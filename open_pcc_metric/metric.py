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

class AbstractPrimaryMetric(abc.ABC):
    value: typing.Any

    @abc.abstractmethod
    def calculate(self, cloud_pair: CloudPair):
        raise NotImplementedError("calculate is not implmented")

class BoundarySqrtDistances(AbstractPrimaryMetric):
    def calculate(self, cloud_pair: CloudPair):
        inner_dists = cloud_pair.origin_cloud.compute_nearest_neighbor_distance()
        self.value = (np.min(inner_dists), np.max(inner_dists))

class GeoMSE(AbstractPrimaryMetric):
    is_left: bool

    def __init__(self, is_left: bool):
        self.is_left = is_left

    def calculate(self, cloud_pair: CloudPair):
        sse = np.sum(cloud_pair._origin_neigh_dists, axis=0)
        n = cloud_pair._origin_neigh_dists.shape[0]
        self.value = sse / n

class ColorMSE(AbstractPrimaryMetric):
    is_left: bool

    def __init__(self, is_left: bool):
        self.is_left = is_left

    def calculate(self, cloud_pair: CloudPair):
        diff = np.subtract(cloud_pair.origin_cloud.colors, cloud_pair._origin_neigh_cloud.colors)
        self.value = np.mean(diff**2, axis=0)

class CalculateResult:
    _metrics: typing.List[AbstractPrimaryMetric]

    def __init__(self, metrics: typing.List[AbstractPrimaryMetric]):
        self._metrics = metrics

    def __str__(self) -> str:
        mnamer = lambda m: str(m.__class__.__name__)
        return "\n".join([mnamer(metric) + " " + str(metric.value) for metric in self._metrics])

class MetricCalculator:
    _metrics: typing.List[AbstractPrimaryMetric]

    def __init__(self, metrics: typing.List[AbstractPrimaryMetric]):
        self._metrics = metrics

    def calculate(self, cloud_pair: CloudPair) -> CalculateResult:
        for metric in self._metrics:
            metric.calculate(cloud_pair)

        return CalculateResult(self._metrics)

def calculate_from_files(
    ocloud_file: str,
    pcloud_file: str,
    ) -> typing.Dict[str, np.float64]:
    ocloud, pcloud = map(o3d.io.read_point_cloud, (ocloud_file, pcloud_file))
    cloud_pair = CloudPair(ocloud, pcloud)
    calculator = MetricCalculator([
        BoundarySqrtDistances(),
        GeoMSE(is_left=True),
        GeoMSE(is_left=False),
        ColorMSE(is_left=True),
        ColorMSE(is_left=False),
    ])

    return calculator.calculate(cloud_pair)
