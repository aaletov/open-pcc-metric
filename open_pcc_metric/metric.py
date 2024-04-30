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
    point: np.array,
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
    for i in range(len(iter_cloud.points)):
        [k, j, dist] = get_ith_nearest(search_cloud.points[i], kdtree, i=n)
        yield (i, j, dist)

def calcullisimmo(
    ocloud: o3d.geometry.PointCloud,
    pcloud: o3d.geometry.PointCloud,
    ) -> typing.Dict[str, np.float64]:
    # Iterate over O, search in P
    ptree = o3d.geometry.KDTreeFlann(pcloud)
    geo_sse_left = 0
    color_sse_left = np.zeros(shape=(3,))

    for [i, j, dist] in nniterator(ocloud, pcloud, ptree, 0):
        geo_sse_left += dist**2
        color_diff = ocloud.colors[i] - pcloud.colors[j]
        color_sse_left += np.power(color_diff, (2, 2, 2))

    # Iterate over P, search in O

    otree = o3d.geometry.KDTreeFlann(ocloud)
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

    return {
        "left_geometric_sse": geo_sse_left,
        "right_geometric_sse": geo_sse_right,
        "left_color_sse": color_sse_left,
        "right_color_sse": color_sse_right,
        "max_geo_dist": max_geo_dist,
    }

def calcullopollo(
    ocloud_file: str,
    pcloud_file: str,
    ) -> typing.Dict[str, np.float64]:
    ocloud, pcloud = map(o3d.io.read_point_cloud, (ocloud_file, pcloud_file))
    return calcullisimmo(ocloud, pcloud)
