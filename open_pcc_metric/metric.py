import typing
import abc
import numpy as np
from numpy.core.multiarray import array as array
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

class CloudPair:
    origin_cloud: o3d.geometry.PointCloud
    reconst_cloud: o3d.geometry.PointCloud
    _origin_tree: typing.Optional[o3d.geometry.KDTreeFlann] = None
    _reconst_tree: typing.Optional[o3d.geometry.KDTreeFlann] = None
    _origin_neigh_cloud: typing.Optional[o3d.geometry.KDTreeFlann] = None
    _origin_neigh_dists : typing.Optional[np.array] = None
    _reconst_neigh_cloud: typing.Optional[o3d.geometry.KDTreeFlann] = None
    _reconst_neigh_dists : typing.Optional[np.array] = None

    def __init__(
        self,
        origin_cloud: o3d.geometry.PointCloud,
        reconst_cloud: o3d.geometry.PointCloud,
    ):
        self.origin_cloud = origin_cloud
        self.reconst_cloud = reconst_cloud
        self._origin_tree = o3d.geometry.KDTreeFlann(self.origin_cloud)
        self._reconst_tree = o3d.geometry.KDTreeFlann(self.reconst_cloud)
        self._origin_neigh_cloud, self._origin_neigh_dists = CloudPair.get_neighbour_cloud(
            iter_cloud=self.origin_cloud,
            search_cloud=self.reconst_cloud,
            kdtree=self._reconst_tree,
            n=0,
        )
        self._reconst_neigh_cloud, self._reconst_neigh_dists = CloudPair.get_neighbour_cloud(
            iter_cloud=self.reconst_cloud,
            search_cloud=self.origin_cloud,
            kdtree=self._origin_tree,
            n=0,
        )

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

    def __str__(self) -> str:
        return "{label}: {value}".format(label=self.label, value=str(self.value))

class PrimaryMetric(AbstractMetric):
    @abc.abstractmethod
    def calculate(self, cloud_pair: CloudPair):
        raise NotImplementedError("calculate is not implmented")

class OrderedMetric(PrimaryMetric):
    is_left: bool

    def __init__(self, is_left: bool):
        self.is_left = is_left

    def __str__(self) -> str:
        order = "left" if self.is_left else "right"
        return "{label}({order}): {value}".format(
            label=self.label,
            order=order,
            value=self.value,
        )

class SecondaryMetric(AbstractMetric):
    def calculate(self, metrics: typing.List[AbstractMetric]) -> bool:
        raise NotImplementedError("calculate is not implmented")

class BoundarySqrtDistances(PrimaryMetric):
    label = "BoundarySqrtDistances"

    def calculate(self, cloud_pair: CloudPair):
        inner_dists = cloud_pair.origin_cloud.compute_nearest_neighbor_distance()
        self.value = (np.min(inner_dists), np.max(inner_dists))

def get_metrics_by_label(
    metrics: typing.List[AbstractMetric],
    label: str,
) -> typing.List[AbstractMetric]:
    return list(filter(lambda m: m.label == label, metrics))

class MinSqrtDistance(SecondaryMetric):
    label = "MinSqrtDistance"

    def calculate(self, metrics: typing.List[AbstractMetric]) -> bool:
        res = get_metrics_by_label(metrics, "BoundarySqrtDistances")
        if len(res) == 0:
            return False
        if len(res) > 1:
            raise RuntimeError("Should not be more than one BoundarySqrtDistance metric")
        boundary_metric = res[0]
        self.value = boundary_metric.value[0]
        return True

class MaxSqrtDistance(SecondaryMetric):
    label = "MaxSqrtDistance"

    def calculate(self, metrics: typing.List[AbstractMetric]) -> bool:
        res = get_metrics_by_label(metrics, "BoundarySqrtDistances")
        if len(res) == 0:
            return False
        if len(res) > 1:
            raise RuntimeError("Should not be more than one BoundarySqrtDistance metric")
        boundary_metric = res[0]
        self.value = boundary_metric.value[1]
        return True

class GeoMSE(OrderedMetric):
    label = "GeoMSE"

    def calculate(self, cloud_pair: CloudPair):
        sse = 0
        if self.is_left:
            sse = np.sum(cloud_pair._origin_neigh_dists, axis=0)
        else:
            sse = np.sum(cloud_pair._reconst_neigh_dists, axis=0)
        n = cloud_pair._origin_neigh_dists.shape[0]
        self.value = sse / n

class ColorMSE(OrderedMetric):
    label = "ColorMSE"

    def calculate(self, cloud_pair: CloudPair):
        diff = 0
        if self.is_left:
            diff = np.subtract(cloud_pair.origin_cloud.colors, cloud_pair._origin_neigh_cloud.colors)
        else:
            diff = np.subtract(cloud_pair.reconst_cloud.colors, cloud_pair._reconst_neigh_cloud.colors)
        self.value = np.mean(diff**2, axis=0)

class SymmetricMetric(SecondaryMetric):
    is_proportional: bool
    target_label: str

    def __init__(self, label: str, is_proportional: bool, target_label: str):
        self.label = label
        self.is_proportional = is_proportional
        self.target_label = target_label

    def calculate(self, metrics: typing.List[AbstractMetric]) -> bool:
        metrics = get_metrics_by_label(metrics, self.target_label)
        if len(metrics) != 2:
            raise RuntimeError("Must be exactly 2 ordered metrics")
        values = [m.value for m in metrics] # value is scalar or ndarray
        if self.is_proportional:
            self.value = min(values, key=np.linalg.norm)
        else:
            self.value = max(values, key=np.linalg.norm)
        return True

    def __str__(self) -> str:
        return "{label}: {value}".format(label=self.label, value=self.value)

class CalculateResult:
    _metrics: typing.List[PrimaryMetric]

    def __init__(self, metrics: typing.List[PrimaryMetric]):
        self._metrics = metrics

    def __str__(self) -> str:
        return "\n".join([str(metric) for metric in self._metrics])

class MetricCalculator:
    _primary_metrics: typing.List[PrimaryMetric]
    _secondary_metrics: typing.List[SecondaryMetric]

    def __init__(
        self,
        primary_metrics: typing.List[PrimaryMetric],
        secondary_metrics: typing.List[SecondaryMetric],
    ):
        self._primary_metrics = primary_metrics
        self._secondary_metrics = secondary_metrics

    def calculate(self, cloud_pair: CloudPair) -> CalculateResult:
        for metric in self._primary_metrics:
            metric.calculate(cloud_pair)

        calculated_metrics = []
        not_calculated_metrics = self._secondary_metrics
        while not_calculated_metrics != []:
            new_not_calculated_metrics = []
            for metric in not_calculated_metrics:
                if metric.calculate(self._primary_metrics + calculated_metrics):
                    calculated_metrics.append(metric)
                else:
                    new_not_calculated_metrics.append(metric)

            not_calculated_metrics = new_not_calculated_metrics

        return CalculateResult(self._primary_metrics + self._secondary_metrics)

def calculate_from_files(
    ocloud_file: str,
    pcloud_file: str,
    ) -> typing.Dict[str, np.float64]:
    ocloud, pcloud = map(o3d.io.read_point_cloud, (ocloud_file, pcloud_file))
    cloud_pair = CloudPair(ocloud, pcloud)
    primary_metrics = [
        BoundarySqrtDistances(),
        GeoMSE(is_left=True),
        GeoMSE(is_left=False),
        ColorMSE(is_left=True),
        ColorMSE(is_left=False),
    ]
    secondary_metrics = [
        MinSqrtDistance(),
        MaxSqrtDistance(),
        SymmetricMetric(
            label="GeoMSE(symmetric)",
            is_proportional=False,
            target_label="GeoMSE",
        ),
        SymmetricMetric(
            label="ColorMSE(symmetric)",
            is_proportional=False,
            target_label="ColorMSE",
        )
    ]
    calculator = MetricCalculator(primary_metrics=primary_metrics, secondary_metrics=secondary_metrics)

    return calculator.calculate(cloud_pair)
