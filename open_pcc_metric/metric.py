import typing
import abc
import numpy as np
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

PointCloud = o3d.geometry.PointCloud
KDFlann = o3d.geometry.KDTreeFlann

class CloudPair:
    clouds: typing.Tuple[PointCloud, PointCloud]
    _trees: typing.Tuple[typing.Optional[KDFlann], typing.Optional[KDFlann]]
    _neigh_clouds: typing.Tuple[PointCloud, PointCloud]
    _neigh_dists: typing.Tuple[typing.Optional[np.ndarray], typing.Optional[np.ndarray]]

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

        origin_neigh_cloud, origin_neigh_dists = CloudPair.get_neighbour_cloud(
            iter_cloud=self.clouds[0],
            search_cloud=self.clouds[1],
            kdtree=self._trees[1],
            n=0,
        )
        reconst_neigh_cloud, reconst_neigh_dists = CloudPair.get_neighbour_cloud(
            iter_cloud=self.clouds[1],
            search_cloud=self.clouds[0],
            kdtree=self._trees[0],
            n=0,
        )
        self._neigh_clouds = (origin_neigh_cloud, reconst_neigh_cloud)
        self._neigh_dists = (origin_neigh_dists, reconst_neigh_dists)

    @staticmethod
    def scale_rgb(cloud: o3d.geometry.PointCloud):
        scaled_colors = np.apply_along_axis(
            func1d=lambda c: 255 * c,
            axis=1,
            arr=cloud.colors
        )
        cloud.colors = o3d.utility.Vector3dVector(scaled_colors)

    @staticmethod
    def convert_cloud_to_yuv(cloud: o3d.geometry.PointCloud):
        """Helper function to convert RGB to YUV (BT.709 or YCoCg-R)
        """
        # only rgb supported
        transform = np.array([
            [0.25, 0.5, 0.25],
            [1, 0, -1],
            [-0.5, 1, -0.5]
        ])
        def converter(c: np.ndarray) -> np.ndarray:
            yuv_c = np.matmul(transform, c)
            # offset 2^8 Co and Cg
            yuv_c[1] += 256
            yuv_c[2] += 256
            return yuv_c

        converted_colors = np.apply_along_axis(
            func1d=converter,
            axis=1,
            arr=cloud.colors
        )
        cloud.colors = o3d.utility.Vector3dVector(converted_colors)

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

class AbstractMetric(abc.ABC):
    label: str
    value: typing.Any

    def __str__(self) -> str:
        return "{label}: {value}".format(label=self.label, value=str(self.value))

    @abc.abstractmethod
    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List['AbstractMetric'],
    ) -> bool:
        raise NotImplementedError("calculate is not implemented")

def get_metrics_by_label(
    metrics: typing.List[AbstractMetric],
    label: str,
) -> typing.List[AbstractMetric]:
    return list(filter(lambda m: m.label == label, metrics))

class DirectionalMetric(AbstractMetric):
    is_left: bool

    def __str__(self) -> str:
        order = "left" if self.is_left else "right"
        return "{label}({order}): {value}".format(
            label=self.label,
            order=order,
            value=self.value,
        )

class PointToPlaneable(DirectionalMetric):
    point_to_plane: bool

    def __str__(self) -> str:
        order = "left" if self.is_left else "right"
        proj_type = "p2plane" if self.point_to_plane else "p2point"
        return "{label}({proj_type})({order}): {value}".format(
            label=self.label,
            proj_type=proj_type,
            order=order,
            value=self.value,
        )

class ErrorVector(PointToPlaneable):
    label = "ErrorVector"
    point_to_plane: bool

    def __init__(self, is_left: bool, point_to_plane: bool):
        self.is_left = is_left
        self.point_to_plane = point_to_plane

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        cloud_idx = 0 if self.is_left else 1
        errs = np.subtract(
            cloud_pair.clouds[cloud_idx].points,
            cloud_pair._neigh_clouds[cloud_idx].points,
        )
        if not self.point_to_plane:
            self.value = errs
        else:
            normals = np.asarray(cloud_pair.clouds[(cloud_idx + 1) % 1].normals)
            plane_errs = np.zeros(shape=(errs.shape[0],))
            for i in range(errs.shape[0]):
                plane_errs[i] = np.dot(errs[i], normals[i])
            self.value = plane_errs
        return True

class EuclideanDistance(PointToPlaneable):
    label = "EuclideanDistance"

    def __init__(self, is_left: bool, point_to_plane: bool):
        self.is_left = is_left
        self.point_to_plane = point_to_plane

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        cloud_idx = 0 if self.is_left else 1
        if not self.point_to_plane:
            self.value = cloud_pair._neigh_dists[cloud_idx]
        else:
            res = get_metrics_by_label(calculated_metrics, "ErrorVector")
            errs = next(filter(lambda m: m.is_left == self.is_left, res), None)
            if errs is None:
                return False
            self.value = np.apply_along_axis(func1d=lambda x: x.dot(x), axis=1, arr=errs.value)
        return True

class BoundarySqrtDistances(AbstractMetric):
    label = "BoundarySqrtDistances"

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        inner_dists = cloud_pair.clouds[0].compute_nearest_neighbor_distance()
        self.value = (np.min(inner_dists), np.max(inner_dists))
        return True

class MinSqrtDistance(AbstractMetric):
    label = "MinSqrtDistance"

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        res = get_metrics_by_label(calculated_metrics, "BoundarySqrtDistances")
        if len(res) == 0:
            return False
        if len(res) > 1:
            raise RuntimeError("Should not be more than one BoundarySqrtDistance metric")
        boundary_metric = res[0]
        self.value = boundary_metric.value[0]
        return True

class MaxSqrtDistance(AbstractMetric):
    label = "MaxSqrtDistance"

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        res = get_metrics_by_label(calculated_metrics, "BoundarySqrtDistances")
        if len(res) == 0:
            return False
        if len(res) > 1:
            raise RuntimeError("Should not be more than one BoundarySqrtDistance metric")
        boundary_metric = res[0]
        self.value = boundary_metric.value[1]
        return True

class GeoMSE(PointToPlaneable):
    label = "GeoMSE"

    def __init__(self, is_left: bool, point_to_plane: bool):
        self.is_left = is_left
        self.point_to_plane = point_to_plane

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        res = get_metrics_by_label(calculated_metrics, "EuclideanDistance")
        pred = lambda m: (m.is_left == self.is_left) and (m.point_to_plane == self.point_to_plane)
        dists = next(filter(pred, res), None)
        if dists is None:
            return False
        n = dists.value.shape[0]
        sse = np.sum(dists.value, axis=0)
        self.value = sse / n
        return True

class GeoPSNR(PointToPlaneable):
    label = "GeoPSNR"

    def __init__(self, is_left: bool, point_to_plane: bool):
        self.is_left = is_left
        self.point_to_plane = point_to_plane

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        peak = None
        if not self.point_to_plane:
            res = get_metrics_by_label(calculated_metrics, "MaxSqrtDistance")
            if len(res) == 0:
                return False
            if len(res) > 1:
                raise RuntimeError("Must be exactly one MaxSqrtDistance metric")
            peak = res[0].value
        else:
            bounding_box: o3d.geometry.OrientedBoundingBox = cloud_pair.clouds[0].get_minimal_oriented_bounding_box()
            peak = np.max(bounding_box.extent)

        res = get_metrics_by_label(calculated_metrics, "GeoMSE")
        pred = lambda m: (m.is_left == self.is_left) and (m.point_to_plane == self.point_to_plane)
        noise = next(filter(pred, res), None)
        if noise is None:
            raise RuntimeError("No corresponding GeoMSE found")
        self.value = 10 * np.log10(peak**2 / noise.value)
        return True

class ColorMSE(DirectionalMetric):
    label = "ColorMSE"

    def __init__(self, is_left: bool):
        self.is_left = is_left

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        cloud_idx = 0 if self.is_left else 1
        diff = np.subtract(
            cloud_pair.clouds[cloud_idx].colors,
            cloud_pair._neigh_clouds[cloud_idx].colors,
        )
        self.value = np.mean(diff**2, axis=0)
        return True

class ColorPSNR(DirectionalMetric):
    label = "ColorPSNR"

    def __init__(self, is_left: bool):
        self.is_left = is_left

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        peak = 255
        res = get_metrics_by_label(calculated_metrics, "ColorMSE")
        noise = next(filter(lambda m: m.is_left == self.is_left, res), None)
        if noise is None:
            raise RuntimeError("No corresponding ColorMSE found")
        self.value = 10 * np.log10(peak**2 / noise.value)
        return True

class GeoHausdorffDistance(PointToPlaneable):
    label = "GeoHausdorffDistance"

    def __init__(self, is_left: bool, point_to_plane: bool):
        self.is_left = is_left
        self.point_to_plane = point_to_plane

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        res = get_metrics_by_label(calculated_metrics, "EuclideanDistance")
        pred = lambda m: (m.is_left == self.is_left) and (m.point_to_plane == self.point_to_plane)
        dists = next(filter(pred, res), None)
        if dists is None:
            return False
        self.value = np.max(dists.value, axis=0)
        return True

class GeoHausdorffDistancePSNR(PointToPlaneable):
    label = "GeoHausdorffDistancePSNR"

    def __init__(self, is_left: bool, point_to_plane: bool):
        self.is_left = is_left
        self.point_to_plane = point_to_plane

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        res = get_metrics_by_label(calculated_metrics, "MaxSqrtDistance")
        if len(res) == 0:
            return False
        if len(res) > 1:
            raise RuntimeError("Must be exactly one MaxSqrtDistance metric")
        peak = res[0]
        res = get_metrics_by_label(calculated_metrics, "GeoHausdorffDistance")
        pred = lambda m: (m.is_left == self.is_left) and (m.point_to_plane == self.point_to_plane)
        hausdorff = next(filter(pred, res), None)
        if hausdorff is None:
            raise RuntimeError("No corresponding GeoHausdorffDistance found")
        self.value = 10 * np.log10(peak.value**2 / hausdorff.value)
        return True

class ColorHausdorffDistance(DirectionalMetric):
    label = "ColorHausdorffDistance"

    def __init__(self, is_left: bool):
        self.is_left = is_left

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        cloud_idx = 0 if self.is_left else 1
        diff = np.subtract(
            cloud_pair.clouds[cloud_idx].colors,
            cloud_pair._neigh_clouds[cloud_idx].colors
        )
        rgb_scale = 255
        self.value = np.max((rgb_scale * diff)**2, axis=0)
        return True

class ColorHausdorffDistancePSNR(DirectionalMetric):
    label = "ColorHausdorffDistancePSNR"

    def __init__(self, is_left: bool):
        self.is_left = is_left

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        peak = 255
        res = get_metrics_by_label(calculated_metrics, "ColorHausdorffDistance")
        hausdorff = next(filter(lambda m: m.is_left == self.is_left, res), None)
        if hausdorff is None:
            return False
        self.value = 10 * np.log10(peak**2 / hausdorff.value)
        return True

class SymmetricMetric(AbstractMetric):
    is_proportional: bool
    predicate: typing.Callable[[AbstractMetric], bool]

    def __init__(
        self,
        label: str,
        is_proportional: bool,
        predicate: typing.Callable[[AbstractMetric], bool],
    ):
        self.label = label
        self.is_proportional = is_proportional
        self.predicate = predicate

    def calculate(
        self,
        cloud_pair: CloudPair,
        calculated_metrics: typing.List[AbstractMetric],
    ) -> bool:
        metrics = list(filter(self.predicate, calculated_metrics))
        if len(metrics) == 0:
            return False
        if len(metrics) > 2:
            raise RuntimeError("Must be exactly 2 ordered metrics for metric %s" % self.label)
        values = [m.value for m in metrics] # value is scalar or ndarray
        if self.is_proportional:
            self.value = min(values, key=np.linalg.norm)
        else:
            self.value = max(values, key=np.linalg.norm)
        return True

    def __str__(self) -> str:
        return "{label}: {value}".format(label=self.label, value=self.value)

    def __repr__(self) -> str:
        return "{label}:{predicate}".format(label=self.label, predicate=self.predicate)

class CalculateResult:
    _metrics: typing.List[AbstractMetric]

    def __init__(self, metrics: typing.List[AbstractMetric]):
        self._metrics = metrics

    def __str__(self) -> str:
        return "\n".join([str(metric) for metric in self._metrics])

class MetricCalculator:
    _metrics: typing.List[AbstractMetric]

    def __init__(
        self,
        metrics: typing.List[AbstractMetric]
    ):
        self._metrics = metrics

    def calculate(self, cloud_pair: CloudPair) -> CalculateResult:
        calculated_metrics = []
        not_calculated_metrics = self._metrics
        new_not_calculated_metrics = []
        while not_calculated_metrics != []:
            if len(not_calculated_metrics) == len(new_not_calculated_metrics):
                raise RuntimeError("Could not calculate metrics: %s" % not_calculated_metrics)
            new_not_calculated_metrics = []
            for metric in not_calculated_metrics:
                if metric.calculate(cloud_pair, calculated_metrics):
                    calculated_metrics.append(metric)
                else:
                    new_not_calculated_metrics.append(metric)

            not_calculated_metrics = new_not_calculated_metrics

        return CalculateResult(calculated_metrics)

def label_predicate(label: str) -> typing.Callable[[AbstractMetric], bool]:
    def pred(metric: AbstractMetric) -> bool:
        return metric.label == label
    return pred

def metric_predicate(
    label: str,
    point_to_plane: typing.Optional[bool] = None,
) -> typing.Callable[[AbstractMetric], bool]:
    def pred(metric: AbstractMetric) -> bool:
        return (
            (metric.label == label) and
            (
                (point_to_plane is None) or
                (metric.point_to_plane == point_to_plane)
            )
        )
    return pred

def calculate_from_files(
    ocloud_file: str,
    pcloud_file: str,
    ) -> typing.Dict[str, np.float64]:
    ocloud, pcloud = map(o3d.io.read_point_cloud, (ocloud_file, pcloud_file))
    cloud_pair = CloudPair(ocloud, pcloud)
    metrics = [
        BoundarySqrtDistances(),
        ErrorVector(is_left=True, point_to_plane=False),
        ErrorVector(is_left=False, point_to_plane=False),
        ErrorVector(is_left=True, point_to_plane=True),
        ErrorVector(is_left=False, point_to_plane=True),
        ColorMSE(is_left=True),
        ColorMSE(is_left=False),
        ColorHausdorffDistance(is_left=True),
        ColorHausdorffDistance(is_left=False),
        MinSqrtDistance(),
        MaxSqrtDistance(),
        EuclideanDistance(is_left=True, point_to_plane=False),
        EuclideanDistance(is_left=False, point_to_plane=False),
        EuclideanDistance(is_left=True, point_to_plane=True),
        EuclideanDistance(is_left=False, point_to_plane=True),
        GeoMSE(is_left=True, point_to_plane=False),
        GeoMSE(is_left=False, point_to_plane=False),
        GeoMSE(is_left=True, point_to_plane=True),
        GeoMSE(is_left=False, point_to_plane=True),
        GeoHausdorffDistance(is_left=True, point_to_plane=False),
        GeoHausdorffDistance(is_left=False, point_to_plane=False),
        GeoHausdorffDistance(is_left=True, point_to_plane=True),
        GeoHausdorffDistance(is_left=False, point_to_plane=True),
        GeoPSNR(is_left=True, point_to_plane=False),
        GeoPSNR(is_left=False, point_to_plane=False),
        GeoPSNR(is_left=True, point_to_plane=True),
        GeoPSNR(is_left=False, point_to_plane=True),
        ColorPSNR(is_left=True),
        ColorPSNR(is_left=False),
        GeoHausdorffDistancePSNR(is_left=True, point_to_plane=False),
        GeoHausdorffDistancePSNR(is_left=False, point_to_plane=False),
        GeoHausdorffDistancePSNR(is_left=True, point_to_plane=True),
        GeoHausdorffDistancePSNR(is_left=False, point_to_plane=True),
        ColorHausdorffDistancePSNR(is_left=True),
        ColorHausdorffDistancePSNR(is_left=False),
        SymmetricMetric(
            label="GeoMSE(p2point)(symmetric)",
            is_proportional=False,
            predicate=metric_predicate("GeoMSE", point_to_plane=False),
        ),
        SymmetricMetric(
            label="GeoMSE(p2plane)(symmetric)",
            is_proportional=False,
            predicate=metric_predicate("GeoMSE", point_to_plane=True),
        ),
        SymmetricMetric(
            label="GeoPSNR(p2point)(symmetric)",
            is_proportional=True,
            predicate=metric_predicate("GeoPSNR", point_to_plane=False),
        ),
        SymmetricMetric(
            label="GeoPSNR(p2plane)(symmetric)",
            is_proportional=True,
            predicate=metric_predicate("GeoPSNR", point_to_plane=True),
        ),
        SymmetricMetric(
            label="ColorMSE(symmetric)",
            is_proportional=False,
            predicate=metric_predicate("ColorMSE"),
        ),
        SymmetricMetric(
            label="ColorPSNR(symmetric)",
            is_proportional=True,
            predicate=metric_predicate("ColorPSNR"),
        ),
        SymmetricMetric(
            label="GeoHausdorffDistance(p2point)(symmetric)",
            is_proportional=False,
            predicate=metric_predicate("GeoHausdorffDistance", point_to_plane=False),
        ),
        SymmetricMetric(
            label="GeoHausdorffDistance(p2plane)(symmetric)",
            is_proportional=False,
            predicate=metric_predicate("GeoHausdorffDistance", point_to_plane=True),
        ),
        SymmetricMetric(
            label="ColorHausdorffDistance(symmetric)",
            is_proportional=False,
            predicate=metric_predicate("ColorHausdorffDistance"),
        ),
        SymmetricMetric(
            label="GeoHausdorffDistancePSNR(p2point)(symmetric)",
            is_proportional=True,
            predicate=metric_predicate("GeoHausdorffDistancePSNR", point_to_plane=False),
        ),
        SymmetricMetric(
            label="GeoHausdorffDistancePSNR(p2plane)(symmetric)",
            is_proportional=True,
            predicate=metric_predicate("GeoHausdorffDistancePSNR", point_to_plane=True),
        ),
        SymmetricMetric(
            label="ColorHausdorffDistancePSNR(symmetric)",
            is_proportional=True,
            predicate=metric_predicate("ColorHausdorffDistancePSNR"),
        ),
    ]
    calculator = MetricCalculator(metrics=metrics)

    return calculator.calculate(cloud_pair)
