import typing
import abc
import numpy as np
import pandas as pd
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
    _calculated_metrics: typing.Dict[typing.Tuple, 'AbstractMetric'] = {}

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
        rpoint = point.reshape((3, 1))
        [k, idx, dists] = kdtree.search_knn_vector_3d(rpoint, n + 1)
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

    def _transform_options(self, options: 'CalculateOptions') -> typing.List['AbstractMetric']:
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
            self.clouds[0].has_colors() and
            self.clouds[1].has_colors() and
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

        return metrics

    def _metric_recursive_calculate(self, metric: 'AbstractMetric') -> 'AbstractMetric':
        if metric._key() in self._calculated_metrics:
            return self._calculated_metrics[metric._key()]

        calculated_deps = {}
        for dep_key, dep_metric in metric._get_dependencies().items():
            calculated_dep_metric = self._metric_recursive_calculate(dep_metric)
            calculated_deps[dep_key] = calculated_dep_metric

        metric.calculate(self, **calculated_deps)
        self._calculated_metrics[metric._key()] = metric

        return metric

    def calculate(self, options: 'CalculateOptions') -> 'CalculateResult':
        metrics_list = self._transform_options(options)

        calculated_metrics_list = []
        for metric in metrics_list:
            calculated_metric = self._metric_recursive_calculate(metric)
            calculated_metrics_list.append(calculated_metric)

        return CalculateResult(calculated_metrics_list)

class AbstractMetric(abc.ABC):
    value: typing.Any
    is_calculated: bool = False

    def _key(self) -> typing.Tuple:
        return (self.__class__.__name__,)

    def _get_dependencies(self) -> typing.Dict[str, 'AbstractMetric']:
        return {}

    @abc.abstractmethod
    def calculate(
        self,
        cloud_pair: CloudPair,
        **kwargs: typing.Dict[str, 'AbstractMetric']
    ) -> None:
        raise NotImplementedError("calculate is not implemented")

    def __str__(self) -> str:
        return "{key}: {value}".format(key=self._key(), value=str(self.value))

class DirectionalMetric(AbstractMetric):
    is_left: bool

    def __init__(self, is_left: bool):
        self.is_left = is_left

    def _key(self) -> typing.Tuple:
        return super()._key() + (self.is_left,)

class PointToPlaneable(DirectionalMetric):
    point_to_plane: bool

    def __init__(self, is_left: bool, point_to_plane: bool):
        super().__init__(is_left)
        self.point_to_plane = point_to_plane

    def _key(self) -> typing.Tuple:
        return super()._key() + (self.point_to_plane,)

class ErrorVector(PointToPlaneable):
    def calculate(
        self,
        cloud_pair: CloudPair,
        **kwargs,
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
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        if not self.point_to_plane:
            return {}

        return {
            "error_vector": ErrorVector(
                is_left=self.is_left,
                point_to_plane=self.point_to_plane,
            )
        }

    def calculate(
        self,
        cloud_pair: CloudPair,
        error_vector: typing.Optional[ErrorVector] = None,
    ) -> None:
        cloud_idx = 0 if self.is_left else 1
        if not self.point_to_plane:
            self.value = cloud_pair._neigh_dists[cloud_idx]
        else:
            self.value = np.square(error_vector.value)
        self.is_calculated = True

class BoundarySqrtDistances(AbstractMetric):
    def calculate(
        self,
        cloud_pair: CloudPair,
        **kwargs,
    ) -> None:
        inner_dists = cloud_pair.clouds[0].compute_nearest_neighbor_distance()
        self.value = (np.min(inner_dists), np.max(inner_dists))
        self.is_calculated = True

class MinSqrtDistance(AbstractMetric):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {"boundary_metric": BoundarySqrtDistances()}

    def calculate(
        self,
        cloud_pair: CloudPair,
        **kwargs,
    ) -> None:
        boundary_metric = BoundarySqrtDistances()
        if not boundary_metric.is_calculated:
            boundary_metric.calculate(cloud_pair)
        self.value = boundary_metric.value[0]
        self.is_calculated = True

class MaxSqrtDistance(AbstractMetric):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {"boundary_metric": BoundarySqrtDistances()}

    def calculate(
        self,
        cloud_pair: CloudPair,
        boundary_metric: BoundarySqrtDistances,
    ) -> None:
        self.value = boundary_metric.value[1]
        self.is_calculated = True

class GeoMSE(PointToPlaneable):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "euclidean_distance": EuclideanDistance(
                is_left=self.is_left,
                point_to_plane=self.point_to_plane,
            )
        }

    def calculate(
        self,
        cloud_pair: CloudPair,
        euclidean_distance: EuclideanDistance,
    ) -> None:
        n = euclidean_distance.value.shape[0]
        sse = np.sum(euclidean_distance.value, axis=0)
        self.value = sse / n
        self.is_calculated = True

class GeoPSNR(PointToPlaneable):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "geo_mse": GeoMSE(
                is_left=self.is_left,
                point_to_plane=self.point_to_plane,
            )
        }

    def calculate(
        self,
        cloud_pair: CloudPair,
        geo_mse: GeoMSE,
    ) -> None:
        bounding_box: o3d.geometry.OrientedBoundingBox = cloud_pair.clouds[0].get_minimal_oriented_bounding_box()
        peak = np.max(bounding_box.extent)
        self.value = 10 * np.log10(peak**2 / geo_mse.value)
        self.is_calculated = True

class ColorMSE(DirectionalMetric):
    def calculate(
        self,
        cloud_pair: CloudPair,
        **kwargs,
    ) -> None:
        cloud_idx = 0 if self.is_left else 1
        diff = np.subtract(
            cloud_pair.clouds[cloud_idx].colors,
            cloud_pair._neigh_clouds[cloud_idx].colors,
        )
        self.value = np.mean(diff**2, axis=0)
        self.is_calculated = True

class ColorPSNR(DirectionalMetric):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {"color_mse": ColorMSE(is_left=self.is_left)}

    def calculate(
        self,
        cloud_pair: CloudPair,
        color_mse: ColorMSE,
    ) -> None:
        peak = 255
        self.value = 10 * np.log10(peak**2 / color_mse.value)
        self.is_calculated = True

class GeoHausdorffDistance(PointToPlaneable):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "euclidean_distance": EuclideanDistance(
                is_left=self.is_left,
                point_to_plane=self.point_to_plane,
            )
        }

    def calculate(
        self,
        cloud_pair: CloudPair,
        euclidean_distance: EuclideanDistance,
    ) -> None:
        self.value = np.max(euclidean_distance.value, axis=0)
        self.is_calculated = True

class GeoHausdorffDistancePSNR(PointToPlaneable):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "max_sqrt": MaxSqrtDistance(),
            "hausdorff_distance": GeoHausdorffDistance(
                is_left=self.is_left,
                point_to_plane=self.point_to_plane,
            ),
        }

    def calculate(
        self,
        cloud_pair: CloudPair,
        max_sqrt: MaxSqrtDistance,
        hausdorff_distance: GeoHausdorffDistance,
    ) -> None:
        self.value = 10 * np.log10(max_sqrt.value**2 / hausdorff_distance.value)
        self.is_calculated = True

class ColorHausdorffDistance(DirectionalMetric):
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
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {"hausdorff_distance": ColorHausdorffDistance(is_left=self.is_left)}

    def calculate(
        self,
        cloud_pair: CloudPair,
        hausdorff_distance: ColorHausdorffDistance,
    ) -> None:
        peak = 255
        self.value = 10 * np.log10(peak**2 / hausdorff_distance.value)
        self.is_calculated = True

class SymmetricMetric(AbstractMetric):
    is_proportional: bool
    metrics: typing.List[DirectionalMetric]

    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "lmetric": self.metrics[0],
            "rmetric": self.metrics[1],
        }

    def __init__(
        self,
        metrics: typing.List[DirectionalMetric],
        is_proportional: bool,
    ):
        if len(metrics) != 2:
            raise ValueError("Must be exactly two metrics")
        if metrics[0].__class__ != metrics[1].__class__:
            raise ValueError(
                "Metrics must be of same class, got: {lmetric}, {rmetric}"
                    .format(lmetric=metrics[0].__class__, rmetric=metrics[1].__class__)
            )
        self.metrics = metrics
        self.is_proportional = is_proportional

    def _key(self) -> typing.Tuple:
        return super()._key() + self.metrics[0]._key() + self.metrics[1]._key()

    def calculate(
        self,
        cloud_pair: CloudPair,
        lmetric: AbstractMetric,
        rmetric: AbstractMetric,
    ) -> None:
        values = [m.value for m in (lmetric, rmetric)] # value is scalar or ndarray
        if self.is_proportional:
            self.value = min(values, key=np.linalg.norm)
        else:
            self.value = max(values, key=np.linalg.norm)
        self.is_calculated = True

    # def __repr__(self) -> str:
    #     return "{label}:{metrics}".format(label=self.label, metrics=self.metrics)

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

    def as_df(self) -> pd.DataFrame:
        # metrics = [str(metric) for metric in self._metrics]
        metric_dict = {
            "label": [],
            "is_left": [],
            "point-to-plane": [],
            "value": [],
        }

        for metric in self._metrics:
            label = metric.__class__.__name__
            if isinstance(metric, SymmetricMetric):
                child_label = metric.metrics[0].__class__.__name__
                label = child_label + "(symmetric)"
            metric_dict["label"].append(label)
            is_left = ""
            if hasattr(metric, "is_left"):
                is_left = metric.is_left
            metric_dict["is_left"].append(is_left)
            point_to_plane = ""
            if hasattr(metric, "point_to_plane"):
                point_to_plane = metric.point_to_plane
            metric_dict["point-to-plane"].append(point_to_plane)
            metric_dict["value"].append(str(metric.value))

        return pd.DataFrame(metric_dict)

    def __str__(self) -> str:
        return str(self.as_df())

# class MetricCalculator:
#     def calculate(
#         self,
#         cloud_pair: CloudPair,
#         options: CalculateOptions,
#     ) -> CalculateResult:


#         print("{num} metrics to calculate".format(num=len(metrics)))

#         for metric in metrics:
#             if not metric.is_calculated:
#                 metric.calculate(cloud_pair)

#         calculated_metrics = list(filter(lambda m: m.is_calculated, metrics))

#         if len(calculated_metrics) != len(metrics):
#             raise RuntimeWarning("Not all metrics were calculated")

#         return CalculateResult(calculated_metrics)

def calculate_from_files(
    ocloud_file: str,
    pcloud_file: str,
    calculate_options: CalculateOptions,
    ) -> pd.DataFrame:
    ocloud, pcloud = map(o3d.io.read_point_cloud, (ocloud_file, pcloud_file))
    cloud_pair = CloudPair(ocloud, pcloud, calculate_options.color)
    return cloud_pair.calculate(calculate_options).as_df()
