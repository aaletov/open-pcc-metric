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

def get_neighbour(
    point: np.ndarray,
    kdtree: o3d.geometry.KDTreeFlann,
    n: int,
) -> np.ndarray:
    [k, idx, dists] = kdtree.search_knn_vector_3d(point, n + 1)
    return np.array((idx[-1], dists[-1]))

def get_neighbour_cloud(
    iter_cloud: o3d.geometry.PointCloud,
    search_cloud: o3d.geometry.PointCloud,
    kdtree: o3d.geometry.KDTreeFlann,
    n: int,
) -> typing.Tuple[o3d.geometry.PointCloud, np.ndarray]:
    finder = lambda point: get_neighbour(point, kdtree, n)
    [idxs, sqrdists] = np.apply_along_axis(finder, axis=1, arr=iter_cloud.points).T
    idxs = idxs.astype(int)
    neigh_points = np.take(search_cloud.points, idxs, axis=0)
    neigh_colors = np.take(search_cloud.colors, idxs, axis=0)
    neigh_cloud = o3d.geometry.PointCloud()
    neigh_cloud.points = o3d.utility.Vector3dVector(neigh_points)
    neigh_cloud.colors = o3d.utility.Vector3dVector(neigh_colors)
    return (neigh_cloud, sqrdists)

class CalculateResult:
    pass

class CloudPair:
    origin_cloud: o3d.geometry.PointCloud
    reconst_cloud: o3d.geometry.PointCloud
    __origin_tree: typing.Optional[o3d.geometry.KDTreeFlann] = None
    __reconst_tree: typing.Optional[o3d.geometry.KDTreeFlann] = None
    __origin_neigh_cloud: typing.Optional[o3d.geometry.KDTreeFlann] = None
    __origin_neigh_dists : typing.Optional[np.array] = None
    __reconst_neigh_cloud: typing.Optional[o3d.geometry.KDTreeFlann] = None
    __reconst_neigh_dists : typing.Optional[np.array] = None

    def __init__(
        self,
        origin_cloud: o3d.geometry.PointCloud,
        reconst_cloud: o3d.geometry.PointCloud,
    ):
        self.origin_cloud = origin_cloud
        self.reconst_cloud = reconst_cloud

    @classmethod
    def get_geo_mse(
        cls,
        sqrdists: np.ndarray,
    ) -> np.float64:
        return np.sum(sqrdists, axis=0) / sqrdists.shape[0]

    @classmethod
    def get_color_mse(
        cls,
        lcloud: o3d.geometry.PointCloud,
        rcloud: o3d.geometry.PointCloud,
    ):
        return np.sum(np.subtract(lcloud.colors, rcloud.colors)**2, axis=0) / len(lcloud.colors)

    @classmethod
    def get_boundary_square_dist(
        cls,
        cloud: o3d.geometry.PointCloud,
    ) -> typing.Tuple[np.float64, np.float64]:
        inner_dists = cloud.compute_nearest_neighbor_distance()
        return (np.min(inner_dists), np.max(inner_dists))

    def calculate(self) -> CalculateResult:
        self.__origin_tree = o3d.geometry.KDTreeFlann(self.origin_cloud)
        self.__reconst_tree = o3d.geometry.KDTreeFlann(self.reconst_cloud)
        self.__origin_neigh_cloud, self.__origin_neigh_dists = get_neighbour_cloud(
            iter_cloud=self.origin_cloud,
            search_cloud=self.reconst_cloud,
            kdtree=self.__reconst_tree,
            n=0,
        )
        self.__reconst_neigh_cloud, self.__reconst_neigh_dists = get_neighbour_cloud(
            iter_cloud=self.reconst_cloud,
            search_cloud=self.origin_cloud,
            kdtree=self.__origin_tree,
            n=0,
        )

        bounds = CloudPair.get_boundary_square_dist(self.origin_cloud)
        return {
            "MinGeoSqrtDist": bounds[0],
            "MaxGeoSqrtDist": bounds[1],
            "GeoMSE": CloudPair.get_geo_mse(self.__origin_neigh_dists),
            "ColorMSE": CloudPair.get_color_mse(self.origin_cloud, self.__origin_neigh_cloud),
        }

def calculate_from_files(
    ocloud_file: str,
    pcloud_file: str,
    ) -> typing.Dict[str, np.float64]:
    ocloud, pcloud = map(o3d.io.read_point_cloud, (ocloud_file, pcloud_file))
    cloud_pair = CloudPair(ocloud, pcloud)
    return cloud_pair.calculate()
