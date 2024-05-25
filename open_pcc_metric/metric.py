import typing
import abc
import sys
import logging
import numpy as np
import pandas as pd
import open3d as o3d

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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
    def get_neighbour(
        point: np.ndarray,
        kdtree: o3d.geometry.KDTreeFlann,
        n: int,
    ) -> np.ndarray:
        rpoint = point.reshape((3, 1))
        [_, idx, dists] = kdtree.search_knn_vector_3d(rpoint, n + 1)
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
                ColorMSE(is_left=True, color_scheme=options.color),
                ColorMSE(is_left=False, color_scheme=options.color),
                SymmetricMetric(
                    metrics=(
                        ColorMSE(is_left=True, color_scheme=options.color),
                        ColorMSE(is_left=False, color_scheme=options.color),
                    ),
                    is_proportional=False,
                ),
                ColorPSNR(is_left=True, color_scheme=options.color),
                ColorPSNR(is_left=False, color_scheme=options.color),
                SymmetricMetric(
                    metrics=(
                        ColorPSNR(is_left=True, color_scheme=options.color),
                        ColorPSNR(is_left=False, color_scheme=options.color),
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
        logger.info("%s metrics to calculate", len(metrics_list))

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

class ColorMetric(DirectionalMetric):
    color_scheme: str

    def __init__(self, is_left: bool, color_scheme: str):
        super().__init__(is_left)
        self.color_scheme = color_scheme

    def _key(self) -> typing.Tuple:
        return super()._key() + (self.color_scheme,)

def transform_colors(
    colors: np.ndarray,
    source_scheme: str,
    target_scheme: str,
) -> np.ndarray:
    if source_scheme == target_scheme:
        return colors

    transform = None
    if (source_scheme == "rgb") and (target_scheme == "ycc"):
        transform = np.array([
            [0.2126, 0.7152, 0.0722],
            [-0.1146, -0.3854, 0.5],
            [0.5, -0.4542, -0.0458],
        ])
    if (source_scheme == "rgb") and (target_scheme == "yuv"):
        transform = np.array([
            [0.25, 0.5, 0.25],
            [1, 0, -1],
            [-0.5, 1, -0.5]
        ])

    def converter(c: np.ndarray) -> np.ndarray:
        return np.matmul(transform , c)

    return np.apply_along_axis(
        func1d=converter,
        axis=1,
        arr=colors,
    )

def get_color_peak(color_scheme: str) -> np.float64:
    colors_to_values = {
        "rgb": 255.0,
        "ycc": 1.0,
        "yuv": 1.0,
    }
    return colors_to_values[color_scheme]


class ColorMSE(ColorMetric):
    def calculate(
        self,
        cloud_pair: CloudPair,
        **kwargs,
    ) -> None:
        cloud_idx = 0 if self.is_left else 1
        orig_colors = np.copy(cloud_pair.clouds[cloud_idx].colors)
        neigh_colors = np.copy(cloud_pair._neigh_clouds[cloud_idx].colors)

        orig_colors = transform_colors(
            colors=orig_colors,
            source_scheme="rgb",
            target_scheme=self.color_scheme,
        )

        neigh_colors = transform_colors(
            colors=neigh_colors,
            source_scheme="rgb",
            target_scheme=self.color_scheme,
        )

        diff = np.subtract(
            orig_colors,
            neigh_colors,
        )
        self.value = np.mean(diff**2, axis=0)
        self.is_calculated = True

class ColorPSNR(ColorMetric):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {"color_mse": ColorMSE(is_left=self.is_left, color_scheme=self.color_scheme)}

    def calculate(
        self,
        cloud_pair: CloudPair,
        color_mse: ColorMSE,
    ) -> None:
        peak = get_color_peak(self.color_scheme)
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

class ColorHausdorffDistance(ColorMetric):
    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        cloud_idx = 0 if self.is_left else 1
        orig_colors = np.copy(cloud_pair.clouds[cloud_idx].colors)
        neigh_colors = np.copy(cloud_pair._neigh_clouds[cloud_idx].colors)

        orig_colors = transform_colors(
            colors=orig_colors,
            source_scheme="rgb",
            target_scheme=self.color_scheme,
        )

        neigh_colors = transform_colors(
            colors=neigh_colors,
            source_scheme="rgb",
            target_scheme=self.color_scheme,
        )

        diff = np.subtract(
            orig_colors,
            neigh_colors,
        )

        # ???
        if self.color_scheme == "rgb":
            rgb_scale = 255
            diff = rgb_scale * diff

        self.value = np.max(diff**2, axis=0)
        self.is_calculated = True

class ColorHausdorffDistancePSNR(ColorMetric):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "hausdorff_distance": ColorHausdorffDistance(
                is_left=self.is_left,
                color_scheme=self.color_scheme,
            ),
        }

    def calculate(
        self,
        cloud_pair: CloudPair,
        hausdorff_distance: ColorHausdorffDistance,
    ) -> None:
        peak = get_color_peak(self.color_scheme)
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

def calculate_from_files(
    ocloud_file: str,
    pcloud_file: str,
    calculate_options: CalculateOptions,
    ) -> pd.DataFrame:
    ocloud, pcloud = map(o3d.io.read_point_cloud, (ocloud_file, pcloud_file))
    cloud_pair = CloudPair(ocloud, pcloud)
    return cloud_pair.calculate(calculate_options).as_df()
