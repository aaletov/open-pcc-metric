import typing
import abc
import numpy as np
import open3d as o3d

# Metrics to calculate:
#   - Geometric PSNR
#       - Geometric pPSNR (max over **original** range)
#       - Geometric MSE (cumulative over both ranges)
#   - RGB PSNR
#       - RGB MSE (cumulative over both ranges)
#
# Both cumulative and max metrics are kind of fold-accumulate

# Maybe work with chunks?

PointCloud = o3d.geometry.PointCloud
KDFlann = o3d.geometry.KDTreeFlann

class CloudPair:
    clouds: typing.Tuple[PointCloud, PointCloud]
    _trees: typing.Tuple[typing.Optional[KDFlann], typing.Optional[KDFlann]]
    _neigh_clouds: typing.Tuple[PointCloud, PointCloud]
    _neigh_dists: typing.Tuple[typing.Optional[np.ndarray], typing.Optional[np.ndarray]]

    def __init__(
        self,
        origin_cloud: o3d.geometry.PointCloud,
        reconst_cloud: o3d.geometry.PointCloud,
        color_scheme: typing.Optional[str] = None,
    ):
        self.clouds = (origin_cloud, reconst_cloud)

        if (
            self.clouds[0].has_colors() and
            self.clouds[1].has_colors() and
            color_scheme == "ycc"
        ):
            print("Converting clouds to ycc")
            CloudPair._convert_cloud_to_ycc(self.clouds[0])
            CloudPair._convert_cloud_to_ycc(self.clouds[1])

        if not self.clouds[0].has_normals():
            self.clouds[0].estimate_normals()
        if not self.clouds[1].has_normals():
            self.clouds[1].estimate_normals()
        self._trees = tuple(map(KDFlann, self.clouds))

        origin_neigh_cloud, origin_neigh_dists = CloudPair.get_neighbour_cloud(
            iter_cloud=self.clouds[0],
            search_cloud=self.clouds[1],
            kdtree=self._trees[1],
            n=0,
        )
        reconst_neigh_cloud, reconst_neigh_dists = CloudPair.get_neighbour_cloud(
            iter_cloud=self.clouds[1],
            search_cloud=self.clouds[0],
            kdtree=self._trees[0],
            n=0,
        )
        self._neigh_clouds = (origin_neigh_cloud, reconst_neigh_cloud)
        self._neigh_dists = (origin_neigh_dists, reconst_neigh_dists)

    @staticmethod
    def _convert_cloud_to_ycc(cloud: o3d.geometry.PointCloud):
        """Helper function to convert RGB to BT.709
        """
        transform = np.array([
            [0.2126, 0.7152, 0.0722],
            [-0.1146, -0.3854, 0.5],
            [0.5, -0.4542, -0.0458],
        ])
        def converter(c: np.ndarray) -> np.ndarray:
            ycc = np.matmul(transform, c)
            return ycc

        converted_colors = np.apply_along_axis(
            func1d=converter,
            axis=1,
            arr=cloud.colors
        )
        cloud.colors = o3d.utility.Vector3dVector(converted_colors)

    @staticmethod
    def _convert_cloud_to_yuv(cloud: o3d.geometry.PointCloud):
        """Helper function to convert RGB to YCoCg-R
        """
        # only rgb supported
        transform = np.array([
            [0.25, 0.5, 0.25],
            [1, 0, -1],
            [-0.5, 1, -0.5]
        ])
        def converter(c: np.ndarray) -> np.ndarray:
            yuv_c = np.matmul(transform, c)
            # offset 2^8 Co and Cg
            yuv_c[1] += 256
            yuv_c[2] += 256
            return yuv_c

        converted_colors = np.apply_along_axis(
            func1d=converter,
            axis=1,
            arr=cloud.colors
        )
        cloud.colors = o3d.utility.Vector3dVector(converted_colors)

    @staticmethod
    def get_neighbour(
        point: np.ndarray,
        kdtree: o3d.geometry.KDTreeFlann,
        n: int,
    ) -> np.ndarray:
        [k, idx, dists] = kdtree.search_knn_vector_3d(point, n + 1)
        return np.array((idx[-1], dists[-1]))

    @staticmethod
    def get_neighbour_cloud(
        iter_cloud: o3d.geometry.PointCloud,
        search_cloud: o3d.geometry.PointCloud,
        kdtree: o3d.geometry.KDTreeFlann,
        n: int,
    ) -> typing.Tuple[o3d.geometry.PointCloud, np.ndarray]:
        finder = lambda point: CloudPair.get_neighbour(point, kdtree, n)
        [idxs, sqrdists] = np.apply_along_axis(finder, axis=1, arr=iter_cloud.points).T
        idxs = idxs.astype(int)
        neigh_points = np.take(search_cloud.points, idxs, axis=0)
        neigh_cloud = o3d.geometry.PointCloud()
        neigh_cloud.points = o3d.utility.Vector3dVector(neigh_points)

        if search_cloud.has_colors():
            neigh_colors = np.take(search_cloud.colors, idxs, axis=0)
            neigh_cloud.colors = o3d.utility.Vector3dVector(neigh_colors)

        return (neigh_cloud, sqrdists)

class AbstractMetric(abc.ABC):
    label: str
    value: typing.Any
    is_calculated: bool = False
    _instances: typing.List['AbstractMetric'] = []

    @classmethod
    def _get_existing(cls, **kwargs) -> typing.Optional['AbstractMetric']:
        for instance in cls._instances:
            if instance._match(**kwargs):
                return instance
        return None

    def _key(self) -> str:
        return "{label}"

    def __str__(self) -> str:
        return "{key}: {value}".format(key=self._key(), value=str(self.value))

    @abc.abstractmethod
    def _match(self, **kwargs) -> bool:
        raise NotImplementedError("raise is not implemented")

    @abc.abstractmethod
    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        raise NotImplementedError("calculate is not implemented")

class SimpleMetric(AbstractMetric):
    def _match(self, **kwargs) -> bool:
        return ("label" in kwargs) and (self.label == kwargs["label"])

def get_metrics_by_label(
    metrics: typing.List[AbstractMetric],
    label: str,
) -> typing.List[AbstractMetric]:
    return list(filter(lambda m: m.label == label, metrics))

class DirectionalMetric(AbstractMetric):
    is_left: bool

    def __new__(cls, is_left: bool) -> 'DirectionalMetric':
        instance = cls._get_existing(label=cls.label, is_left=is_left)
        if instance is None:
            new_instance = super(DirectionalMetric, cls).__new__(cls)
            new_instance.is_left = is_left
            cls._instances.append(new_instance)
            return new_instance
        return instance

    def _match(self, **kwargs) -> bool:
        return (
            ("label" in kwargs) and
            (self.label == kwargs["label"]) and
            ("is_left" in kwargs) and
            (self.is_left == kwargs["is_left"])
        )

    def _key(self) -> str:
        order = "left" if self.is_left else "right"
        return "{label}({order})".format(label=self.label, order=order)

class PointToPlaneable(DirectionalMetric):
    point_to_plane: bool

    def __new__(cls, is_left: bool, point_to_plane: bool) -> 'PointToPlaneable':
        instance = cls._get_existing(
            label=cls.label,
            is_left=is_left,
            point_to_plane=point_to_plane,
        )
        if instance is None:
            new_instance = super(PointToPlaneable, cls).__new__(cls, is_left)
            new_instance.is_left = is_left
            new_instance.point_to_plane = point_to_plane
            cls._instances.append(new_instance)
            return new_instance
        return instance

    def _match(self, **kwargs) -> bool:
        return (
            ("label" in kwargs) and
            (self.label == kwargs["label"]) and
            ("is_left" in kwargs) and
            (self.is_left == kwargs["is_left"]) and
            ("point_to_plane" in kwargs) and
            (self.point_to_plane == kwargs["point_to_plane"])
        )

    def _key(self) -> str:
        order = "left" if self.is_left else "right"
        proj_type = "p2plane" if self.point_to_plane else "p2point"
        return "{label}({proj_type})({order})".format(
            label=self.label,
            proj_type=proj_type,
            order=order,
        )

class ErrorVector(PointToPlaneable):
    label = "ErrorVector"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        cloud_idx = 0 if self.is_left else 1
        errs = np.subtract(
            cloud_pair.clouds[cloud_idx].points,
            cloud_pair._neigh_clouds[cloud_idx].points,
        )
        if not self.point_to_plane:
            self.value = errs
        else:
            normals = np.asarray(cloud_pair.clouds[(cloud_idx + 1) % 1].normals)
            plane_errs = np.zeros(shape=(errs.shape[0],))
            for i in range(errs.shape[0]):
                plane_errs[i] = np.dot(errs[i], normals[i])
            self.value = plane_errs
        self.is_calculated = True

class EuclideanDistance(PointToPlaneable):
    label = "EuclideanDistance"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        cloud_idx = 0 if self.is_left else 1
        if not self.point_to_plane:
            self.value = cloud_pair._neigh_dists[cloud_idx]
        else:
            errs = ErrorVector(
                is_left=self.is_left,
                point_to_plane=self.point_to_plane,
            )
            if not errs.is_calculated:
                errs.calculate(cloud_pair)
            self.value = np.square(errs.value)
        self.is_calculated = True

class BoundarySqrtDistances(SimpleMetric):
    label = "BoundarySqrtDistances"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        inner_dists = cloud_pair.clouds[0].compute_nearest_neighbor_distance()
        self.value = (np.min(inner_dists), np.max(inner_dists))
        self.is_calculated = True

class MinSqrtDistance(SimpleMetric):
    label = "MinSqrtDistance"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        boundary_metric = BoundarySqrtDistances()
        if not boundary_metric.is_calculated:
            boundary_metric.calculate(cloud_pair)
        self.value = boundary_metric.value[0]
        self.is_calculated = True

class MaxSqrtDistance(SimpleMetric):
    label = "MaxSqrtDistance"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        boundary_metric = BoundarySqrtDistances()
        if not boundary_metric.is_calculated:
            boundary_metric.calculate(cloud_pair)
        self.value = boundary_metric.value[1]
        self.is_calculated = True

class GeoMSE(PointToPlaneable):
    label = "GeoMSE"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        dists = EuclideanDistance(
            is_left=self.is_left,
            point_to_plane=self.point_to_plane,
        )
        if not dists.is_calculated:
            dists.calculate(cloud_pair)
        n = dists.value.shape[0]
        sse = np.sum(dists.value, axis=0)
        self.value = sse / n
        self.is_calculated = True

class GeoPSNR(PointToPlaneable):
    label = "GeoPSNR"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        bounding_box: o3d.geometry.OrientedBoundingBox = cloud_pair.clouds[0].get_minimal_oriented_bounding_box()
        peak = np.max(bounding_box.extent)

        geo_mse = GeoMSE(
            is_left=self.is_left,
            point_to_plane=self.point_to_plane,
        )

        if not geo_mse.is_calculated:
            geo_mse.calculate(cloud_pair)

        self.value = 10 * np.log10(peak**2 / geo_mse.value)
        self.is_calculated = True

class ColorMSE(DirectionalMetric):
    label = "ColorMSE"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        cloud_idx = 0 if self.is_left else 1
        diff = np.subtract(
            cloud_pair.clouds[cloud_idx].colors,
            cloud_pair._neigh_clouds[cloud_idx].colors,
        )
        self.value = np.mean(diff**2, axis=0)
        self.is_calculated = True

class ColorPSNR(DirectionalMetric):
    label = "ColorPSNR"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        peak = 255
        color_mse = ColorMSE(is_left=self.is_left)
        if not color_mse.is_calculated:
            color_mse.calculate(cloud_pair)

        self.value = 10 * np.log10(peak**2 / color_mse.value)
        self.is_calculated = True

class GeoHausdorffDistance(PointToPlaneable):
    label = "GeoHausdorffDistance"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        dists = EuclideanDistance(
            is_left=self.is_left,
            point_to_plane=self.point_to_plane,
        )
        if not dists.is_calculated:
            dists.calculate(cloud_pair)
        self.value = np.max(dists.value, axis=0)
        self.is_calculated = True

class GeoHausdorffDistancePSNR(PointToPlaneable):
    label = "GeoHausdorffDistancePSNR"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        max_sqrt = MaxSqrtDistance()
        if not max_sqrt.is_calculated:
            max_sqrt.calculate(cloud_pair)

        hausdorff = GeoHausdorffDistance(
            is_left=self.is_left,
            point_to_plane=self.point_to_plane,
        )
        if not hausdorff.is_calculated:
            hausdorff.calculate(cloud_pair)

        self.value = 10 * np.log10(max_sqrt.value**2 / hausdorff.value)
        self.is_calculated = True

class ColorHausdorffDistance(DirectionalMetric):
    label = "ColorHausdorffDistance"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        cloud_idx = 0 if self.is_left else 1
        diff = np.subtract(
            cloud_pair.clouds[cloud_idx].colors,
            cloud_pair._neigh_clouds[cloud_idx].colors
        )
        rgb_scale = 255
        self.value = np.max((rgb_scale * diff)**2, axis=0)
        self.is_calculated = True

class ColorHausdorffDistancePSNR(DirectionalMetric):
    label = "ColorHausdorffDistancePSNR"

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        peak = 255
        hausdorff = ColorHausdorffDistance(
            is_left=self.is_left,
        )
        if not hausdorff.is_calculated:
            hausdorff.calculate(cloud_pair)
        self.value = 10 * np.log10(peak**2 / hausdorff.value)
        self.is_calculated = True

class SymmetricMetric(SimpleMetric):
    is_proportional: bool
    metrics: typing.List[DirectionalMetric]

    def __init__(
        self,
        metrics: typing.List[DirectionalMetric],
        is_proportional: bool,
    ):
        if len(metrics) != 2:
            raise ValueError("Must be exactly two metrics")
        if metrics[0].label != metrics[1].label:
            raise ValueError(
                "Metrics must be of same class, got: {lmetric}, {rmetric}"
                    .format(lmetric=metrics[0].label, rmetric=metrics[1].label)
            )
        self.label = metrics[0].label + "(symmetric)"
        self.metrics = metrics
        self.is_proportional = is_proportional

    def _key(self) -> str:
        if hasattr(self.metrics[0], "point_to_plane"):
            return "{label}({p2plane})(symmetric)".format(
                label=self.metrics[0].label,
                p2plane="p2plane" if self.metrics[0].point_to_plane else "p2point",
            )
        return self.label

    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        for metric in self.metrics:
            if not metric.is_calculated:
                metric.calculate(cloud_pair)
        values = [m.value for m in self.metrics] # value is scalar or ndarray
        if self.is_proportional:
            self.value = min(values, key=np.linalg.norm)
        else:
            self.value = max(values, key=np.linalg.norm)
        self.is_calculated = True

    def __repr__(self) -> str:
        return "{label}:{metrics}".format(label=self.label, metrics=self.metrics)

class CalculateOptions:
    color: typing.Optional[str]
    hausdorff: bool
    point_to_plane: bool

    def __init__(
        self,
        color: typing.Optional[str] = None,
        hausdorff: bool = False,
        point_to_plane: bool = False,
    ):
        self.color = color
        self.hausdorff = hausdorff
        self.point_to_plane = point_to_plane

class CalculateResult:
    _metrics: typing.List[AbstractMetric]

    def __init__(self, metrics: typing.List[AbstractMetric]):
        self._metrics = metrics

    def as_dict(self) -> typing.Dict[str, typing.Any]:
        d = dict()
        for metric in self._metrics:
            d[metric._key()] = metric.value
        return d

    def __str__(self) -> str:
        return "\n".join([str(metric) for metric in self._metrics])

class MetricCalculator:
    def calculate(
        self,
        cloud_pair: CloudPair,
        options: CalculateOptions,
    ) -> CalculateResult:
        metrics = [
            MinSqrtDistance(),
            MaxSqrtDistance(),
            GeoMSE(is_left=True, point_to_plane=False),
            GeoMSE(is_left=False, point_to_plane=False),
            SymmetricMetric(
                metrics=(
                    GeoMSE(is_left=True, point_to_plane=False),
                    GeoMSE(is_left=False, point_to_plane=False),
                ),
                is_proportional=False,
            ),
            GeoPSNR(is_left=True, point_to_plane=False),
            GeoPSNR(is_left=False, point_to_plane=False),
            SymmetricMetric(
                metrics=(
                    GeoPSNR(is_left=True, point_to_plane=False),
                    GeoPSNR(is_left=False, point_to_plane=False),
                ),
                is_proportional=True,
            ),
        ]

        if (
            cloud_pair.clouds[0].has_colors() and
            cloud_pair.clouds[1].has_colors() and
            (options.color is not None)
        ):
            metrics += [
                ColorMSE(is_left=True),
                ColorMSE(is_left=False),
                SymmetricMetric(
                    metrics=(
                        ColorMSE(is_left=True),
                        ColorMSE(is_left=False),
                    ),
                    is_proportional=False,
                ),
                ColorPSNR(is_left=True),
                ColorPSNR(is_left=False),
                SymmetricMetric(
                    metrics=(
                        ColorPSNR(is_left=True),
                        ColorPSNR(is_left=False),
                    ),
                    is_proportional=True,
                ),
            ]

        if options.point_to_plane:
            metrics += [
                GeoMSE(is_left=True, point_to_plane=True),
                GeoMSE(is_left=False, point_to_plane=True),
                SymmetricMetric(
                    metrics=(
                        GeoMSE(is_left=True, point_to_plane=True),
                        GeoMSE(is_left=False, point_to_plane=True),
                    ),
                    is_proportional=False,
                ),
                GeoPSNR(is_left=True, point_to_plane=True),
                GeoPSNR(is_left=False, point_to_plane=True),
                SymmetricMetric(
                    metrics=(
                        GeoPSNR(is_left=True, point_to_plane=True),
                        GeoPSNR(is_left=False, point_to_plane=True),
                    ),
                    is_proportional=True,
                ),
            ]

        if options.hausdorff:
            metrics += [
                GeoHausdorffDistance(is_left=True, point_to_plane=False),
                GeoHausdorffDistance(is_left=False, point_to_plane=False),
                SymmetricMetric(
                    metrics=(
                        GeoHausdorffDistance(is_left=True, point_to_plane=False),
                        GeoHausdorffDistance(is_left=False, point_to_plane=False),
                    ),
                    is_proportional=False,
                ),
                GeoHausdorffDistancePSNR(is_left=True, point_to_plane=False),
                GeoHausdorffDistancePSNR(is_left=False, point_to_plane=False),
                SymmetricMetric(
                    metrics=(
                        GeoHausdorffDistancePSNR(is_left=True, point_to_plane=False),
                        GeoHausdorffDistancePSNR(is_left=False, point_to_plane=False),
                    ),
                    is_proportional=True,
                ),
            ]

        if options.hausdorff and options.point_to_plane:
            metrics += [
                GeoHausdorffDistance(is_left=True, point_to_plane=True),
                GeoHausdorffDistance(is_left=False, point_to_plane=True),
                GeoHausdorffDistancePSNR(is_left=True, point_to_plane=True),
                GeoHausdorffDistancePSNR(is_left=False, point_to_plane=True),
                SymmetricMetric(
                    metrics=(
                        GeoHausdorffDistance(is_left=True, point_to_plane=True),
                        GeoHausdorffDistance(is_left=False, point_to_plane=True),
                    ),
                    is_proportional=False,
                ),
                SymmetricMetric(
                    metrics=(
                        GeoHausdorffDistancePSNR(is_left=True, point_to_plane=True),
                        GeoHausdorffDistancePSNR(is_left=False, point_to_plane=True),
                    ),
                    is_proportional=True,
                ),
            ]

        print("{num} metrics to calculate".format(num=len(metrics)))

        for metric in metrics:
            if not metric.is_calculated:
                metric.calculate(cloud_pair)

        calculated_metrics = list(filter(lambda m: m.is_calculated, metrics))

        if len(calculated_metrics) != len(metrics):
            raise RuntimeWarning("Not all metrics were calculated")

        return CalculateResult(calculated_metrics)

def calculate_from_files(
    ocloud_file: str,
    pcloud_file: str,
    calculate_options: CalculateOptions,
    ) -> typing.Dict[str, np.float64]:
    ocloud, pcloud = map(o3d.io.read_point_cloud, (ocloud_file, pcloud_file))
    cloud_pair = CloudPair(ocloud, pcloud, calculate_options.color)
    calculator = MetricCalculator()
    return calculator.calculate(cloud_pair, calculate_options)
