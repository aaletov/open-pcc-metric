import typing
import abc
import numpy as np
import open3d as o3d
from .logger import get_logger
from .cloud_pair import CloudPair

logger = get_logger()

PointCloud = o3d.geometry.PointCloud
KDFlann = o3d.geometry.KDTreeFlann


class AbstractMetric(abc.ABC):
    value: typing.Any

    def _key(self) -> typing.Tuple:
        return (self.__class__.__name__,)

    @abc.abstractmethod
    def calculate(
        self,
        cloud_pair: CloudPair,
        **kwargs: typing.Dict[str, 'AbstractMetric']
    ) -> None:
        raise NotImplementedError("calculate is not implemented")

    def __str__(self) -> str:
        return "{key}: {value}".format(key=self._key(), value=str(self.value))


class PrimaryMetric(AbstractMetric):
    @abc.abstractmethod
    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        raise NotImplementedError("calculate is not implemented")


class SecondaryMetric(AbstractMetric):
    def _get_dependencies(self) -> typing.Dict[str, 'AbstractMetric']:
        return {}

    @abc.abstractmethod
    def calculate(
        self,
        **kwargs: typing.Dict[str, 'AbstractMetric']
    ) -> None:
        raise NotImplementedError("calculate is not implemented")


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


class PrimaryErrorVector(PrimaryMetric, DirectionalMetric):
    def calculate(self, cloud_pair: CloudPair) -> None:
        if self.is_left:
            self.value = cloud_pair.get_left_error_vector()
            return

        self.value = cloud_pair.get_right_error_vector()


class NeighbourDistances(PrimaryMetric, DirectionalMetric):
    def calculate(self, cloud_pair: CloudPair) -> None:
        if self.is_left:
            self.value = cloud_pair.get_left_neighbour_distances()
            return

        self.value = cloud_pair.get_right_neighbour_distances()


class CloudNormals(PrimaryMetric, DirectionalMetric):
    def calculate(self, cloud_pair: CloudPair) -> None:
        if self.is_left:
            self.value = np.asarray(cloud_pair.clouds[0].normals)
            return

        self.value = np.asarray(cloud_pair.clouds[1].normals)


class CloudExtent(PrimaryMetric):
    def calculate(self, cloud_pair: CloudPair) -> None:
        self.value = cloud_pair.get_extent()


class CloudColors(PrimaryMetric, DirectionalMetric):
    def calculate(self, cloud_pair: CloudPair) -> None:
        if self.is_left:
            self.value = cloud_pair.get_left_colors()
            return

        self.value = cloud_pair.get_right_colors()


class NeighbourColors(PrimaryMetric, DirectionalMetric):
    def calculate(self, cloud_pair: CloudPair) -> None:
        if self.is_left:
            self.value = cloud_pair.get_left_neighbour_colors()
            return

        self.value = cloud_pair.get_right_neighbour_colors()


class ErrorVector(SecondaryMetric, PointToPlaneable):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        deps = {
            "primary_error_vector": PrimaryErrorVector(is_left=self.is_left)
        }
        if self.point_to_plane:
            deps["cloud_normals"] = CloudNormals(is_left=not self.is_left)
        return deps

    def calculate(
        self,
        primary_error_vector: PrimaryErrorVector,
        cloud_normals: typing.Optional[CloudNormals] = None,
    ) -> None:
        if not self.point_to_plane:
            self.value = np.apply_along_axis(
                func1d=np.linalg.norm,
                axis=1,
                arr=primary_error_vector.value,
            )
            return

        error_shape = primary_error_vector.value.shape
        plane_errs = np.zeros(shape=(error_shape[0],))
        for i in range(error_shape[0]):
            plane_errs[i] = np.dot(
                primary_error_vector.value[i],
                cloud_normals.value[i],
            )
        self.value = plane_errs


class EuclideanDistance(SecondaryMetric, PointToPlaneable):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        if not self.point_to_plane:
            return {
                "neighbour_distances": NeighbourDistances(is_left=self.is_left)
            }

        return {
            "error_vector": ErrorVector(
                is_left=self.is_left,
                point_to_plane=self.point_to_plane,
            )
        }

    def calculate(
        self,
        neighbour_distances: typing.Optional[NeighbourDistances] = None,
        error_vector: typing.Optional[ErrorVector] = None,
    ) -> None:
        if not self.point_to_plane:
            self.value = neighbour_distances.value
            return

        self.value = np.square(error_vector.value)


class BoundarySqrtDistances(PrimaryMetric):
    def calculate(
        self,
        cloud_pair: CloudPair,
    ) -> None:
        inner_dists = cloud_pair.get_boundary_sqrt_distances()
        self.value = (np.min(inner_dists), np.max(inner_dists))


class MinSqrtDistance(SecondaryMetric):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {"boundary_metric": BoundarySqrtDistances()}

    def calculate(
        self,
        boundary_metric: BoundarySqrtDistances,
    ) -> None:
        self.value = boundary_metric.value[0]


class MaxSqrtDistance(SecondaryMetric):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {"boundary_metric": BoundarySqrtDistances()}

    def calculate(
        self,
        boundary_metric: BoundarySqrtDistances,
    ) -> None:
        self.value = boundary_metric.value[1]


class GeoMSE(SecondaryMetric, PointToPlaneable):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "euclidean_distance": EuclideanDistance(
                is_left=self.is_left,
                point_to_plane=self.point_to_plane,
            )
        }

    def calculate(
        self,
        euclidean_distance: EuclideanDistance,
    ) -> None:
        n = euclidean_distance.value.shape[0]
        sse = np.sum(euclidean_distance.value, axis=0)
        self.value = sse / n


class GeoPSNR(SecondaryMetric, PointToPlaneable):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "cloud_extent": CloudExtent(),
            "geo_mse": GeoMSE(
                is_left=self.is_left,
                point_to_plane=self.point_to_plane,
            ),
        }

    def calculate(
        self,
        cloud_extent: CloudExtent,
        geo_mse: GeoMSE,
    ) -> None:
        peak = np.max(cloud_extent.value)
        self.value = 10 * np.log10(peak**2 / geo_mse.value)


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


class ColorMSE(SecondaryMetric, ColorMetric):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "origin_cloud_colors": CloudColors(is_left=self.is_left),
            "neighbour_cloud_colors": NeighbourColors(is_left=self.is_left),
        }

    def calculate(
        self,
        origin_cloud_colors: CloudColors,
        neighbour_cloud_colors: NeighbourColors,
    ) -> None:
        orig_colors = np.copy(origin_cloud_colors.value)
        neigh_colors = np.copy(neighbour_cloud_colors.value)

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


class ColorPSNR(SecondaryMetric, ColorMetric):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "color_mse": ColorMSE(
                is_left=self.is_left,
                color_scheme=self.color_scheme,
            ),
        }

    def calculate(
        self,
        color_mse: ColorMSE,
    ) -> None:
        peak = get_color_peak(self.color_scheme)
        self.value = 10 * np.log10(peak**2 / color_mse.value)


class GeoHausdorffDistance(SecondaryMetric, PointToPlaneable):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "euclidean_distance": EuclideanDistance(
                is_left=self.is_left,
                point_to_plane=self.point_to_plane,
            )
        }

    def calculate(
        self,
        euclidean_distance: EuclideanDistance,
    ) -> None:
        self.value = np.max(euclidean_distance.value, axis=0)


class GeoHausdorffDistancePSNR(SecondaryMetric, PointToPlaneable):
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
        max_sqrt: MaxSqrtDistance,
        hausdorff_distance: GeoHausdorffDistance,
    ) -> None:
        self.value = 10 * np.log10(
            max_sqrt.value**2 / hausdorff_distance.value,
        )


class ColorHausdorffDistance(SecondaryMetric, ColorMetric):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "origin_cloud_colors": CloudColors(is_left=self.is_left),
            "neighbour_cloud_colors": NeighbourColors(is_left=self.is_left),
        }

    def calculate(
        self,
        origin_cloud_colors: CloudColors,
        neighbour_cloud_colors: NeighbourColors,
    ) -> None:
        orig_colors = np.copy(origin_cloud_colors.value)
        neigh_colors = np.copy(neighbour_cloud_colors.value)

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


class ColorHausdorffDistancePSNR(SecondaryMetric, ColorMetric):
    def _get_dependencies(self) -> typing.Dict[str, AbstractMetric]:
        return {
            "hausdorff_distance": ColorHausdorffDistance(
                is_left=self.is_left,
                color_scheme=self.color_scheme,
            ),
        }

    def calculate(
        self,
        hausdorff_distance: ColorHausdorffDistance,
    ) -> None:
        peak = get_color_peak(self.color_scheme)
        self.value = 10 * np.log10(peak**2 / hausdorff_distance.value)


class SymmetricMetric(SecondaryMetric):
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
            lclass = metrics[0].__class__
            rclass = metrics[1].__class__
            raise ValueError(
                f"Metrics must be of same class, got: {lclass}, {rclass}"
            )
        self.metrics = metrics
        self.is_proportional = is_proportional

    def _key(self) -> typing.Tuple:
        return super()._key() + self.metrics[0]._key() + self.metrics[1]._key()

    def calculate(
        self,
        lmetric: AbstractMetric,
        rmetric: AbstractMetric,
    ) -> None:
        # value is scalar or ndarray
        values = [m.value for m in (lmetric, rmetric)]
        if self.is_proportional:
            self.value = min(values, key=np.linalg.norm)
        else:
            self.value = max(values, key=np.linalg.norm)
