import pytest
import open3d as o3d
import numpy as np
import open_pcc_metric.metric as opmm

default_nrows = 3


@pytest.fixture()
def default_cloud_pair() -> opmm.CloudPair:
    origin_points = np.eye(default_nrows, dtype="float64")
    origin_colors = np.copy(origin_points)
    err = 1e-1 * np.linspace(1.0, default_nrows, default_nrows)
    reconst_points = origin_points + err
    reconst_colors = origin_colors + err
    origin_cloud = o3d.geometry.PointCloud()
    reconst_cloud = o3d.geometry.PointCloud()
    origin_cloud.points = o3d.utility.Vector3dVector(origin_points)
    origin_cloud.colors = o3d.utility.Vector3dVector(origin_colors)
    reconst_cloud.points = o3d.utility.Vector3dVector(reconst_points)
    reconst_cloud.colors = o3d.utility.Vector3dVector(reconst_colors)
    return opmm.CloudPair(origin_cloud, reconst_cloud)


# add point_to_plane (how to setup normals?)
@pytest.mark.parametrize(
    "is_left,point_to_plane",
    [(True, False), (False, False)],
)
def test_default_error_vector(
    is_left: bool,
    point_to_plane: bool,
):
    error_vector = opmm.ErrorVector(
        is_left=is_left,
        point_to_plane=point_to_plane,
    )
    if not point_to_plane:
        primary_error_vector = opmm.PrimaryErrorVector(is_left=is_left)
        primary_error_vector.value = np.ones(shape=(5, 3), dtype="float64")
        error_vector.calculate(primary_error_vector)
        expected_error_value = np.sqrt(3) * np.ones(shape=(5,))
        assert (np.allclose(error_vector.value, expected_error_value))
    else:
        assert (True)


# add point_to_plane (how to setup normals?)
@pytest.mark.parametrize(
    "is_left,point_to_plane",
    [(True, False), (False, False), (True, True), (False, True)],
)
def test_default_euclidean_distance(
    is_left: bool,
    point_to_plane: bool,
):
    euclidean_distance = opmm.EuclideanDistance(
        is_left=is_left,
        point_to_plane=point_to_plane,
    )
    primary_error_vector = opmm.PrimaryErrorVector(is_left=is_left)
    primary_error_vector.value = 2 * np.ones(shape=(5,))
    neighbour_distances = opmm.NeighbourDistances(is_left=is_left)
    neighbour_distances.value = 4 * np.ones(shape=(5,))
    euclidean_distance.calculate(neighbour_distances, primary_error_vector)
    assert (np.allclose(neighbour_distances.value, euclidean_distance.value))


def test_default_boundary_sqrt_distance(default_cloud_pair: opmm.CloudPair):
    assert (True)


# add point_to_plane (how to setup normals?)
@pytest.mark.parametrize(
    "is_left,point_to_plane",
    [(True, False), (False, False)],
)
def test_default_geo_mse(
    default_cloud_pair: opmm.CloudPair,
    is_left: bool,
    point_to_plane: bool,
):
    assert (True)


# add point_to_plane (how to setup normals?)
@pytest.mark.parametrize(
    "is_left,point_to_plane",
    [(True, False), (False, False)],
)
def test_default_geo_psnr(
    default_cloud_pair: opmm.CloudPair,
    is_left: bool,
    point_to_plane: bool,
):
    assert (True)
