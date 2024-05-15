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

class NumPyPointCloud:
    points: np.ndarray[typing.Any, typing.Any]
    colors: np.ndarray[typing.Any, typing.Any]

    def __init__(
        self,
        points: np.ndarray[typing.Any, typing.Any],
        colors: np.ndarray[typing.Any, typing.Any],
    ):
        self.points = points
        self.colors = colors

class NumPyNeighbourCloud:
    _origin_cloud: NumPyPointCloud
    cloud: NumPyPointCloud # neighbour cloud
    sqrdists: np.ndarray

    def __init__(
        self,
        origin_cloud: NumPyPointCloud,
        cloud: NumPyPointCloud,
        sqrdists: np.ndarray,
    ):
        self._origin_cloud = origin_cloud
        self.cloud = cloud
        self.sqrdists = sqrdists

def read_numpy_point_cloud(cloud: o3d.geometry.PointCloud) -> NumPyPointCloud:
    return NumPyPointCloud(
        points=np.asarray(cloud.points),
        colors=np.asarray(cloud.colors),
    )

class AbstractFoldMetric(abc.ABC):
    @abc.abstractmethod
    def add_points(
        self,
        origin_cloud: NumPyPointCloud,
        neighbour_cloud: NumPyNeighbourCloud,
    ) -> None:
        raise NotImplementedError()

# class MaxGeoSqrDist:
#     LABEL = "MaxGeoSqrDist"
#     value: np.float64 = np.finfo(np.float64).min

# class MinGeoSqrDist:
#     LABEL = "MinGeoSqrDist"
#     value: np.float64 = np.finfo(np.float64).max

class GeoSSE(AbstractFoldMetric):
    LABEL = "GeoSSE"
    value: np.float64 = 0
    num: int = 0

    def add_points(
        self,
        origin_cloud: NumPyPointCloud,
        neighbour_cloud: NumPyNeighbourCloud,
    ) -> None:
        self.value += np.sum(neighbour_cloud.sqrdists, axis=0)
        self.num += neighbour_cloud.sqrdists.shape[0]

class ColorSSE(AbstractFoldMetric):
    LABEL = "ColorSSE"
    value: np.ndarray[typing.Any, typing.Any] = np.zeros(shape=(3,))
    num: int = 0

    def add_points(
        self,
        origin_cloud: NumPyPointCloud,
        neighbour_cloud: NumPyNeighbourCloud,
    ) -> None:
        err = neighbour_cloud.cloud.colors - origin_cloud.colors
        self.value += np.sum(np.power(err, 2), axis=0)
        self.num += err.shape[0]

def get_neighbour(
    point: np.ndarray,
    kdtree: o3d.geometry.KDTreeFlann,
    n: int,
) -> np.ndarray:
    [k, idx, dists] = kdtree.search_knn_vector_3d(point, n + 1)
    return np.array((idx[-1], dists[-1]))

def get_neighbour_cloud(
    iter_cloud: NumPyPointCloud,
    search_cloud: NumPyPointCloud,
    kdtree: o3d.geometry.KDTreeFlann,
    n: int,
) -> NumPyNeighbourCloud:
    finder = lambda point: get_neighbour(point, kdtree, n)
    [idxs, sqrdists] = np.apply_along_axis(finder, axis=1, arr=iter_cloud.points).T
    idxs = idxs.astype(int)
    neighbour_cloud = NumPyPointCloud(
        points=np.take(search_cloud.points, indices=idxs, axis=0),
        colors=np.take(search_cloud.colors, indices=idxs, axis=0),
    )
    return NumPyNeighbourCloud(
        origin_cloud=iter_cloud,
        cloud=neighbour_cloud,
        sqrdists=sqrdists,
    )

class SimpleMetric:
    LABEL: str = ""
    value: typing.Any

    def __init__(self, value: typing.Any, label: str):
        self.value = value
        self.LABEL = label

def get_boundary_square_dist(
    cloud: o3d.geometry.PointCloud,
) -> typing.Tuple[np.float64, np.float64]:
    inner_dists = cloud.compute_nearest_neighbor_distance()
    return (
        SimpleMetric(np.min(inner_dists), "MinGeoSqrtDist"),
        SimpleMetric(np.max(inner_dists), "MaxGeoSqrtDist"),
    )

def calculate(
    origin_cloud: o3d.geometry.PointCloud,
    processed_cloud: o3d.geometry.PointCloud,
    ) -> typing.Dict[str, np.float64]:
    np_origin_cloud = read_numpy_point_cloud(origin_cloud)
    np_processed_cloud = read_numpy_point_cloud(processed_cloud)

    # Iterate over O, search in P
    ptree = o3d.geometry.KDTreeFlann(processed_cloud)
    lneighbours = get_neighbour_cloud(
        iter_cloud=np_origin_cloud,
        search_cloud=np_processed_cloud,
        kdtree=ptree,
        n=0,
    )
    lmetrics = (GeoSSE(), ColorSSE())
    for metric in lmetrics:
        metric.add_points(origin_cloud=origin_cloud, neighbour_cloud=lneighbours)

    # Iterate over P, search in O

    otree = o3d.geometry.KDTreeFlann(origin_cloud)
    rneighbours = get_neighbour_cloud(
        iter_cloud=processed_cloud,
        search_cloud=origin_cloud,
        kdtree=otree,
        n=0,
    )
    rmetrics = (GeoSSE(), ColorSSE())

    for metric in rmetrics:
        metric.add_points(origin_cloud=processed_cloud, neighbour_cloud=rneighbours)

    # Iterate over O, search in O
    self_metrics = get_boundary_square_dist(origin_cloud)
    return self_metrics + lmetrics + rmetrics

def calculate_from_files(
    ocloud_file: str,
    pcloud_file: str,
    ) -> typing.Dict[str, np.float64]:
    ocloud, pcloud = map(o3d.io.read_point_cloud, (ocloud_file, pcloud_file))
    return calculate(ocloud, pcloud)
