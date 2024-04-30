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

def get_ith_nearest(
    point: np.ndarray,
    kdtree: o3d.geometry.KDTreeFlann,
    i: int,
    ) -> typing.Tuple[int, int, np.float64]:
    [k, idx, dist] = kdtree.search_knn_vector_3d(point, i + 1)
    return (k, idx[-1], dist[-1])

def nniterator(
    iter_cloud: o3d.geometry.PointCloud,
    search_cloud: o3d.geometry.PointCloud,
    kdtree: o3d.geometry.KDTreeFlann,
    n: int,
) -> typing.Generator[typing.Tuple[int, int, np.float64], None, None]:
    points = np.asarray(iter_cloud.points)
    for i in range(points.shape[0]):
        [k, j, dist] = get_ith_nearest(points[i], kdtree, i=n)
        yield (i, j, dist)

def calcullisimmo(
    ocloud: o3d.geometry.PointCloud,
    pcloud: o3d.geometry.PointCloud,
    ) -> typing.Dict[str, np.float64]:
    # Iterate over O, search in P
    ptree = o3d.geometry.KDTreeFlann(pcloud)
    geo_sse_left = 0
    points_num_left = 0
    color_sse_left = np.zeros(shape=(3,))

    for [i, j, dist] in nniterator(ocloud, pcloud, ptree, 0):
        geo_sse_left += dist
        points_num_left += 1
        color_diff = ocloud.colors[i] - pcloud.colors[j]
        color_sse_left += np.power(color_diff, (2, 2, 2))

    # Iterate over P, search in O

    otree = o3d.geometry.KDTreeFlann(ocloud)
    geo_sse_right = 0
    points_num_right = 0
    color_sse_right = np.zeros(shape=(3,))

    for [i, j, dist] in nniterator(pcloud, ocloud, otree, 0):
        geo_sse_right += dist
        points_num_right += 1
        color_diff = pcloud.colors[i] - ocloud.colors[j]
        color_sse_right += np.power(color_diff, (2, 2, 2))

    # Iterate over O, search in O
    max_geo_sqrdist = np.finfo(np.float64).min
    min_geo_sqrdist = np.finfo(np.float64).max

    for [i, j, sqrdist] in nniterator(ocloud, ocloud, otree, 1):
        if sqrdist > max_geo_sqrdist:
            max_geo_sqrdist = sqrdist

        if sqrdist < min_geo_sqrdist:
            min_geo_sqrdist = sqrdist

    return {
        "left_geometric_mse": geo_sse_left / points_num_left,
        "right_geometric_mse": geo_sse_right / points_num_right,
        "left_color_mse": color_sse_left / points_num_left,
        "right_color_mse": color_sse_right / points_num_right,
        "max_geo_dist": np.sqrt(max_geo_sqrdist),
    }

def calcullopollo(
    ocloud_file: str,
    pcloud_file: str,
    ) -> typing.Dict[str, np.float64]:
    ocloud, pcloud = map(o3d.io.read_point_cloud, (ocloud_file, pcloud_file))
    return calcullisimmo(ocloud, pcloud)
