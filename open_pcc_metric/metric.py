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
    ):
        self.clouds = (origin_cloud, reconst_cloud)
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
    def scale_rgb(cloud: o3d.geometry.PointCloud):
        scaled_colors = np.apply_along_axis(
            func1d=lambda c: 255 * c,
            axis=1,
            arr=cloud.colors
        )
        cloud.colors = o3d.utility.Vector3dVector(scaled_colors)

    @staticmethod
    def convert_cloud_to_yuv(cloud: o3d.geometry.PointCloud):
        """Helper function to convert RGB to YUV (BT.709 or YCoCg-R)
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
        neigh_colors = np.take(search_cloud.colors, idxs, axis=0)
        neigh_cloud = o3d.geometry.PointCloud()
        neigh_cloud.points = o3d.utility.Vector3dVector(neigh_points)
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

    def __str__(self) -> str:
        return "{label}: {value}".format(label=self.label, value=str(self.value))

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

    def __str__(self) -> str:
        order = "left" if self.is_left else "right"
        return "{label}({order}): {value}".format(
            label=self.label,
            order=order,
            value=self.value,
        )

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

    def __str__(self) -> str:
        order = "left" if self.is_left else "right"
        proj_type = "p2plane" if self.point_to_plane else "p2point"
        return "{label}({proj_type})({order}): {value}".format(
            label=self.label,
            proj_type=proj_type,
            order=order,
            value=self.value,
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
        peak = None
        if not self.point_to_plane:
            max_sqrt = MaxSqrtDistance()
            if not max_sqrt.is_calculated:
                max_sqrt.calculate(cloud_pair)
            peak = max_sqrt.value
        else:
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

    def __str__(self) -> str:
        return "{label}: {value}".format(label=self.label, value=self.value)

    def __repr__(self) -> str:
        return "{label}:{metrics}".format(label=self.label, metrics=self.metrics)

class CalculateResult:
    _metrics: typing.List[AbstractMetric]

    def __init__(self, metrics: typing.List[AbstractMetric]):
        self._metrics = metrics

    def __str__(self) -> str:
        return "\n".join([str(metric) for metric in self._metrics])

class MetricCalculator:
    _metrics: typing.List[AbstractMetric]

    def __init__(
        self,
        metrics: typing.List[AbstractMetric]
    ):
        self._metrics = metrics

    def calculate(self, cloud_pair: CloudPair) -> CalculateResult:
        for metric in self._metrics:
            if not metric.is_calculated:
                metric.calculate(cloud_pair)

        calculated_metrics = list(filter(lambda m: m.is_calculated, self._metrics))
        return CalculateResult(calculated_metrics)

def calculate_from_files(
    ocloud_file: str,
    pcloud_file: str,
    ) -> typing.Dict[str, np.float64]:
    ocloud, pcloud = map(o3d.io.read_point_cloud, (ocloud_file, pcloud_file))
    cloud_pair = CloudPair(ocloud, pcloud)
    metrics = [
        BoundarySqrtDistances(),
        ErrorVector(is_left=True, point_to_plane=False),
        ErrorVector(is_left=False, point_to_plane=False),
        ErrorVector(is_left=True, point_to_plane=True),
        ErrorVector(is_left=False, point_to_plane=True),
        ColorMSE(is_left=True),
        ColorMSE(is_left=False),
        ColorHausdorffDistance(is_left=True),
        ColorHausdorffDistance(is_left=False),
        MinSqrtDistance(),
        MaxSqrtDistance(),
        EuclideanDistance(is_left=True, point_to_plane=False),
        EuclideanDistance(is_left=False, point_to_plane=False),
        EuclideanDistance(is_left=True, point_to_plane=True),
        EuclideanDistance(is_left=False, point_to_plane=True),
        GeoMSE(is_left=True, point_to_plane=False),
        GeoMSE(is_left=False, point_to_plane=False),
        GeoMSE(is_left=True, point_to_plane=True),
        GeoMSE(is_left=False, point_to_plane=True),
        GeoHausdorffDistance(is_left=True, point_to_plane=False),
        GeoHausdorffDistance(is_left=False, point_to_plane=False),
        GeoHausdorffDistance(is_left=True, point_to_plane=True),
        GeoHausdorffDistance(is_left=False, point_to_plane=True),
        GeoPSNR(is_left=True, point_to_plane=False),
        GeoPSNR(is_left=False, point_to_plane=False),
        GeoPSNR(is_left=True, point_to_plane=True),
        GeoPSNR(is_left=False, point_to_plane=True),
        ColorPSNR(is_left=True),
        ColorPSNR(is_left=False),
        GeoHausdorffDistancePSNR(is_left=True, point_to_plane=False),
        GeoHausdorffDistancePSNR(is_left=False, point_to_plane=False),
        GeoHausdorffDistancePSNR(is_left=True, point_to_plane=True),
        GeoHausdorffDistancePSNR(is_left=False, point_to_plane=True),
        ColorHausdorffDistancePSNR(is_left=True),
        ColorHausdorffDistancePSNR(is_left=False),
        SymmetricMetric(
            metrics=(
                GeoMSE(is_left=True, point_to_plane=False),
                GeoMSE(is_left=False, point_to_plane=False),
            ),
            is_proportional=False,
        ),
        SymmetricMetric(
            metrics=(
                GeoMSE(is_left=True, point_to_plane=True),
                GeoMSE(is_left=False, point_to_plane=True),
            ),
            is_proportional=False,
        ),
        SymmetricMetric(
            metrics=(
                GeoPSNR(is_left=True, point_to_plane=False),
                GeoPSNR(is_left=False, point_to_plane=False),
            ),
            is_proportional=True,
        ),
        SymmetricMetric(
            metrics=(
                GeoPSNR(is_left=True, point_to_plane=True),
                GeoPSNR(is_left=False, point_to_plane=True),
            ),
            is_proportional=True,
        ),
        SymmetricMetric(
            metrics=(
                ColorMSE(is_left=True),
                ColorMSE(is_left=False),
            ),
            is_proportional=False,
        ),
        SymmetricMetric(
            metrics=(
                ColorPSNR(is_left=True),
                ColorPSNR(is_left=False),
            ),
            is_proportional=True,
        ),
        SymmetricMetric(
            metrics=(
                GeoHausdorffDistance(is_left=True, point_to_plane=False),
                GeoHausdorffDistance(is_left=False, point_to_plane=False),
            ),
            is_proportional=False,
        ),
        SymmetricMetric(
            metrics=(
                GeoHausdorffDistance(is_left=True, point_to_plane=True),
                GeoHausdorffDistance(is_left=False, point_to_plane=True),
            ),
            is_proportional=False,
        ),
        SymmetricMetric(
            metrics=(
                ColorHausdorffDistance(is_left=True),
                ColorHausdorffDistance(is_left=False),
            ),
            is_proportional=False,
        ),
        SymmetricMetric(
            metrics=(
                GeoHausdorffDistancePSNR(is_left=True, point_to_plane=False),
                GeoHausdorffDistancePSNR(is_left=False, point_to_plane=False),
            ),
            is_proportional=True,
        ),
        SymmetricMetric(
            metrics=(
                GeoHausdorffDistancePSNR(is_left=True, point_to_plane=True),
                GeoHausdorffDistancePSNR(is_left=False, point_to_plane=True),
            ),
            is_proportional=True,
        ),
        SymmetricMetric(
            metrics=(
                ColorHausdorffDistancePSNR(is_left=True),
                ColorHausdorffDistancePSNR(is_left=False),
            ),
            is_proportional=True,
        ),
    ]
    print("{num} metrics to calculate".format(num=len(metrics)))
    calculator = MetricCalculator(metrics=metrics)

    return calculator.calculate(cloud_pair)
