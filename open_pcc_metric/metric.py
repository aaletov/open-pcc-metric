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

class AbstractFoldMetric(abc.ABC):
    @abc.abstractmethod
    def add_point(self, opoint: np.array, ppoint: np.array, dist: np.float64) -> None:
        raise NotImplementedError()

class MaxGeoDistance(AbstractFoldMetric):
    value: np.float64 = np.finfo(np.float64).min

    def add_point(self, opoint: np.array, ppoint: np.array, dist: np.float64) -> None:
        if dist > value:
            value = dist

class MinGeoDistance(AbstractFoldMetric):
    value: np.float64 = np.finfo(np.float64).max

    def add_point(self, opoint: np.array, ppoint: np.array, dist: np.float64) -> None:
        if dist < value:
            value = dist

def get_ith_nearest(
    point: np.array,
    kdtree: o3d.geometry.KDFlann,
    i: int,
    ) -> typing.Tuple[int, int, np.float64]:
    [k, idx, dist] = kdtree.search_knn_vector_3d(point, i + 1)
    return (k[-1], idx[-1], dist[-1])

def nniterator(
    iter_cloud: o3d.geometry.PointCloud,
    search_cloud: o3d.geometry.PointCloud,
    kdtree: o3d.geometry.KDFlann,
    n: int,
) -> typing.Generator[typing.Tuple[int, int, np.float64], None, None]:
    for i in range(len(iter_cloud.point.shape[0])):
        [k, j, dist] = get_ith_nearest(search_cloud.points[i], kdtree, i=n)
        yield (i, j, dist)

def calcullisimmo(ocloud: o3d.geometry.PointCloud, pcloud: o3d.geometry.PointCloud) -> None:
    # Iterate over O, search in P
    ptree = o3d.geometry.KDFlann(pcloud)
    geo_sse_left = 0
    color_sse_left = np.zeros(shape=(3,))

    for [i, j, dist] in nniterator(ocloud, pcloud, ptree, 0):
        geo_sse_left += dist**2
        color_diff = ocloud.colors[i] - pcloud.colors[j]
        color_sse_left += np.power(color_diff, (2, 2, 2))

    # Iterate over P, search in O

    otree = o3d.geometry.KDFlann(ocloud)
    geo_sse_right = 0
    color_sse_right = np.zeros(shape=(3,))

    for [i, j, dist] in nniterator(pcloud, ocloud, otree, 0):
        geo_sse_right += dist**2
        color_diff = pcloud.colors[i] - ocloud.colors[j]
        color_sse_right += np.power(color_diff, (2, 2, 2))

    # Iterate over O, search in O
    max_geo_dist = np.finfo(np.float64).min
    min_geo_dist = np.finfo(np.float64).max

    for [i, j, dist] in nniterator(ocloud, ocloud, otree, 1):
        if dist > max_geo_dist:
            max_geo_dist = dist

        if dist < min_geo_dist:
            min_geo_dist = dist




# Single responsibility violated
class PccMetrics:
    max_geometric_distance_NN: np.float64 = 0
    min_geometric_distance_NN: np.float64 = 0
    geometric_mse: np.float64 = 0
    color_mse: np.float64 = 0

    def __init__(self):
        return

    def add_point(self, opoint: np.array, ppoint: np.array) -> None:
        pass


def get_PSNR(mse: np.float64, peak: np.float64) -> np.float64:
    return 10 * np.log10(peak ** 2 / mse)

def get_pPSNR(cloud: o3d.geometry.PointCloud, kdtree: o3d.geometry.KDFlann) -> np.float64:
    max_dist = np.finfo(np.float64).min
    min_dist = np.finfo(np.float64).max
    for point in cloud.points:
        [k, idx, dist] = kdtree.search_knn_vector_3d(point, 2)

        if dist[-1] > max_dist:
            max_dist = dist[-1]

        if dist[-1] < min_dist:
            min_dist = dist[-1]

    return max_dist

def get_MSE(
    ocloud: o3d.geometry.PointCloud,
    otree: o3d.geometry.KDFlann,
    pcloud: o3d.geometry.PointCloud,
) -> np.float64:
    sum_error = 0
    n = pcloud.points.shape()[0]

    for i in range(n):
        [k, idx, dist] = otree.search_knn_vector_3d(pcloud.points[i], 1)
        sum_error += dist[0]**2

    return sum_error / n



def calculate(ocloud: o3d.geometry.PointCloud, pcloud: o3d.geometry.PointCloud) -> None:
    """Calculates distortion error between original point cloud and processed
    point cloud

    Args:
        ocloud (o3d.geometry.PointCloud): Original point cloud
        pcloud (o3d.geometry.PointCloud): Processed point cloud
    """
    otree = o3d.geometry.KDFlann(ocloud)
    [k, idx, dist] = otree.search_knn_vector_3d(pcloud.points[0], 1)

    k  = typing.cast(int, k)
    idx = typing.cast(o3d.utility.IntVector, idx)
    # In the sources this variable has name "distance2" - it is 2'd norm?
    dist = typing.cast(o3d.utility.DoubleVector, dist)

    error = dist**2

