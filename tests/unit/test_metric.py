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
    default_cloud_pair: opmm.CloudPair,
    is_left: bool,
    point_to_plane: bool,
):
    error_vector = opmm.ErrorVector(is_left=is_left, point_to_plane=point_to_plane)
    error_vector.calculate(default_cloud_pair)
    assert(
        all([
            np.allclose(
                np.absolute(vector),
                1e-1 * np.linspace(
                    1.0,
                    default_nrows,
                    default_nrows,
                ),
            ) for vector in error_vector.value
        ])
    )

# add point_to_plane (how to setup normals?)
@pytest.mark.parametrize(
    "is_left,point_to_plane",
    [(True, False), (False, False)],
)
def test_default_euclidean_distance(
    default_cloud_pair: opmm.CloudPair,
    is_left: bool,
    point_to_plane: bool,
):
    assert(True)

def test_default_boundary_sqrt_distance(default_cloud_pair: opmm.CloudPair):
    assert(True)

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
    assert(True)

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
    assert(True)