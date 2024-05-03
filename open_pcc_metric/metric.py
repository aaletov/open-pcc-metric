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

class AbstractFoldMetric(abc.ABC):
    @abc.abstractmethod
    def add_point(
        self,
        lpoint: np.ndarray[typing.Any, typing.Any],
        rpoint: np.ndarray[typing.Any, typing.Any],
        sqrdist: np.float64,
    ) -> None:
        raise NotImplementedError()

class MaxGeoSqrDist(AbstractFoldMetric):
    LABEL = "MaxGeoSqrDist"
    value: np.float64 = np.finfo(np.float64).min

    def add_point(
        self,
        lpoint: np.ndarray[typing.Any, typing.Any],
        rpoint: np.ndarray[typing.Any, typing.Any],
        sqrdist: np.floating[np.float64],
        ) -> None:
        if sqrdist > self.value:
            self.value = sqrdist

class MinGeoSqrDist(AbstractFoldMetric):
    LABEL = "MinGeoSqrDist"
    value: np.float64 = np.finfo(np.float64).max

    def add_point(
        self,
        lpoint: np.ndarray[typing.Any, typing.Any],
        rpoint: np.ndarray[typing.Any, typing.Any],
        sqrdist: np.floating[np.float64],
        ) -> None:
        if sqrdist < self.value:
            self.value = sqrdist

class GeoSSE(AbstractFoldMetric):
    LABEL = "GeoSSE"
    value: np.float64 = 0
    num: int = 0

    def add_point(
        self,
        lpoint: np.ndarray[typing.Any, typing.Any],
        rpoint: np.ndarray[typing.Any, typing.Any],
        sqrdist: np.floating[np.float64],
        ) -> None:
        self.value += sqrdist
        self.num += 1

class ColorSSE(AbstractFoldMetric):
    LABEL = "ColorSSE"
    value: np.ndarray[typing.Any, typing.Any] = np.zeros(shape=(3,))
    num: int = 0

    def add_point(
        self,
        lpoint: np.ndarray[typing.Any, typing.Any],
        rpoint: np.ndarray[typing.Any, typing.Any],
        sqrdist: np.floating[np.float64],
        ) -> None:
        # i = 1 stores color vector
        err = lpoint[1] - rpoint[1]
        self.value += np.power(err, (2, 2, 2))
        self.num += 1

def nniterator(
    iter_cloud: o3d.geometry.PointCloud,
    search_cloud: o3d.geometry.PointCloud,
    kdtree: o3d.geometry.KDTreeFlann,
    n: int,
) -> typing.Generator[typing.Tuple[int, int, np.float64], None, None]:
    """_summary_

    Args:
        iter_cloud (o3d.geometry.PointCloud): A cloud to iterate over
        search_cloud (o3d.geometry.PointCloud): A cloud for search
        kdtree (o3d.geometry.KDTreeFlann): KDTreeFlann of search_cloud
        n (int): Order of neighbour to search (0 - point itself)

    Returns:
        typing.Generator[typing.Tuple[int, int, np.float64], None, None]: _description_

    Yields:
        Iterator[typing.Generator[typing.Tuple[int, int, np.float64], None, None]]: _description_
    """
    ipoints = np.asarray(iter_cloud.points)
    spoints = np.asarray(search_cloud.points)
    for i in range(ipoints.shape[0]):
        [k, idx, dists] = kdtree.search_knn_vector_3d(ipoints[i], n + 1)
        j = idx[-1]
        dist = dists[-1]
        lpoint = np.stack([ipoints[i], iter_cloud.colors[i]])
        rpoint = np.stack([spoints[j], search_cloud.colors[j]])
        yield (lpoint, rpoint, dist)

def calcullisimmo(
    ocloud: o3d.geometry.PointCloud,
    pcloud: o3d.geometry.PointCloud,
    ) -> typing.Dict[str, np.float64]:
    # Iterate over O, search in P

    ptree = o3d.geometry.KDTreeFlann(pcloud)
    lmetrics = (GeoSSE(), ColorSSE())

    for [lpoint, rpoint, dist] in nniterator(ocloud, pcloud, ptree, 0):
        for metric in lmetrics:
            metric.add_point(lpoint, rpoint, dist)

    # Iterate over P, search in O

    otree = o3d.geometry.KDTreeFlann(ocloud)
    rmetrics = (GeoSSE(), ColorSSE())

    for [lpoint, rpoint, dist] in nniterator(pcloud, ocloud, otree, 0):
        for metric in rmetrics:
            metric.add_point(lpoint, rpoint, dist)

    # Iterate over O, search in O
    self_metrics = (MaxGeoSqrDist(), MinGeoSqrDist())

    for [lpoint, rpoint, sqrdist] in nniterator(ocloud, ocloud, otree, 1):
        for metric in self_metrics:
            metric.add_point(lpoint, rpoint, sqrdist)

    return self_metrics + lmetrics + rmetrics

def calcullopollo(
    ocloud_file: str,
    pcloud_file: str,
    ) -> typing.Dict[str, np.float64]:
    ocloud, pcloud = map(o3d.io.read_point_cloud, (ocloud_file, pcloud_file))
    return calcullisimmo(ocloud, pcloud)
