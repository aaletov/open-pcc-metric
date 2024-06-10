import typing
import numpy as np
import open3d as o3d


PointCloud = o3d.geometry.PointCloud
KDFlann = o3d.geometry.KDTreeFlann


def get_neighbour_cloud(
    iter_cloud: o3d.geometry.PointCloud,
    search_cloud: o3d.geometry.PointCloud,
    kdtree: o3d.geometry.KDTreeFlann,
    n: int,
) -> typing.Tuple[o3d.geometry.PointCloud, np.ndarray]:
    def get_neighbour(
        point: np.ndarray,
        kdtree: o3d.geometry.KDTreeFlann,
        n: int,
    ) -> np.ndarray:
        rpoint = point.reshape((3, 1))
        [_, idx, dists] = kdtree.search_knn_vector_3d(rpoint, n + 1)
        return np.array((idx[-1], dists[-1]))

    def finder(point: np.ndarray):
        return get_neighbour(point, kdtree, n)

    [idxs, sqrdists] = np.apply_along_axis(
        finder,
        axis=1,
        arr=iter_cloud.points,
    ).T
    idxs = idxs.astype(int)
    neigh_points = np.take(search_cloud.points, idxs, axis=0)
    neigh_cloud = o3d.geometry.PointCloud()
    neigh_cloud.points = o3d.utility.Vector3dVector(neigh_points)

    if search_cloud.has_colors():
        neigh_colors = np.take(search_cloud.colors, idxs, axis=0)
        neigh_cloud.colors = o3d.utility.Vector3dVector(neigh_colors)

    return (neigh_cloud, sqrdists)


class CloudPair:
    clouds: typing.Tuple[PointCloud, PointCloud]
    _trees: typing.Tuple[typing.Optional[KDFlann], typing.Optional[KDFlann]]
    _neigh_clouds: typing.Tuple[PointCloud, PointCloud]
    _neigh_dists: typing.Tuple[
        typing.Optional[np.ndarray],
        typing.Optional[np.ndarray],
    ]

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

        origin_neigh_cloud, origin_neigh_dists = get_neighbour_cloud(
            iter_cloud=self.clouds[0],
            search_cloud=self.clouds[1],
            kdtree=self._trees[1],
            n=0,
        )
        reconst_neigh_cloud, reconst_neigh_dists = get_neighbour_cloud(
            iter_cloud=self.clouds[1],
            search_cloud=self.clouds[0],
            kdtree=self._trees[0],
            n=0,
        )
        self._neigh_clouds = (origin_neigh_cloud, reconst_neigh_cloud)
        self._neigh_dists = (origin_neigh_dists, reconst_neigh_dists)

    @property
    def origin_cloud(self):
        return self.clouds[0]

    @property
    def reconst_cloud(self):
        return self.clouds[1]

    def get_left_error_vector(self):
        return np.subtract(
            self.clouds[0].points,
            self._neigh_clouds[0].points,
        )

    def get_right_error_vector(self):
        return np.subtract(
            self.clouds[1].points,
            self._neigh_clouds[1].points,
        )

    def get_left_neighbour_distances(self):
        return self._neigh_dists[0]

    def get_right_neighbour_distances(self):
        return self._neigh_dists[1]

    def get_boundary_sqrt_distances(self):
        return self.clouds[0].compute_nearest_neighbor_distance()

    def get_extent(self):
        return self.clouds[0].get_minimal_oriented_bounding_box().extent

    def get_left_colors(self):
        return self.clouds[0].colors

    def get_right_colors(self):
        return self.clouds[1].colors

    def get_left_neighbour_colors(self):
        return self._neigh_clouds[0].colors

    def get_right_neighbour_colors(self):
        return self._neigh_clouds[1].colors
