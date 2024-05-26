import typing
import pandas as pd
from .metric import (
    AbstractMetric,
    SymmetricMetric,
    PrimaryMetric,
    SecondaryMetric,
)
from .logger import get_logger
from .cloud_pair import CloudPair

logger = get_logger()


class CalculateResult:
    _metrics: typing.List[AbstractMetric]

    def __init__(self, metrics: typing.List[AbstractMetric]):
        self._metrics = metrics

    def as_dict(self) -> typing.Dict[str, typing.Any]:
        d = dict()
        for metric in self._metrics:
            d[metric._key()] = metric.value
        return d

    def as_df(self) -> pd.DataFrame:
        # metrics = [str(metric) for metric in self._metrics]
        metric_dict = {
            "label": [],
            "is_left": [],
            "point-to-plane": [],
            "value": [],
        }

        for metric in self._metrics:
            label = metric.__class__.__name__
            if isinstance(metric, SymmetricMetric):
                child_label = metric.metrics[0].__class__.__name__
                label = child_label + "(symmetric)"
            metric_dict["label"].append(label)
            is_left = ""
            if hasattr(metric, "is_left"):
                is_left = metric.is_left
            metric_dict["is_left"].append(is_left)
            point_to_plane = ""
            if hasattr(metric, "point_to_plane"):
                point_to_plane = metric.point_to_plane
            metric_dict["point-to-plane"].append(point_to_plane)
            metric_dict["value"].append(str(metric.value))

        return pd.DataFrame(metric_dict)

    def __str__(self) -> str:
        return str(self.as_df())


class MetricCalculator:
    _cloud_pair: CloudPair
    _calculated_metrics: typing.Dict[typing.Tuple, 'AbstractMetric'] = {}

    def __init__(self, cloud_pair: CloudPair):
        self._cloud_pair = cloud_pair

    def _metric_recursive_calculate(
        self,
        metric: 'AbstractMetric',
    ) -> 'AbstractMetric':
        if metric._key() in self._calculated_metrics:
            return self._calculated_metrics[metric._key()]

        if not isinstance(metric, (PrimaryMetric, SecondaryMetric)):
            # Would this work?
            unknwn = metric.__class__.__name__
            raise RuntimeError(
                f"Metric of unknown AbstractMetric subclass {unknwn}",
            )

        if isinstance(metric, PrimaryMetric):
            metric = typing.cast(PrimaryMetric, metric)
            metric.calculate(self._cloud_pair)
            self._calculated_metrics[metric._key()] = metric
            return metric

        calculated_deps = {}
        for dep_key, dep_metric in metric._get_dependencies().items():
            calculated_dep_metric = self._metric_recursive_calculate(
                metric=dep_metric,
            )
            calculated_deps[dep_key] = calculated_dep_metric

        metric.calculate(**calculated_deps)
        self._calculated_metrics[metric._key()] = metric

        return metric

    def calculate(
        self,
        metrics_list: typing.List[AbstractMetric],
    ) -> CalculateResult:
        # logger.info("%s metrics to calculate", len(metrics_list))

        calculated_metrics_list = []
        for metric in metrics_list:
            calculated_metric = self._metric_recursive_calculate(metric)
            calculated_metrics_list.append(calculated_metric)

        return CalculateResult(calculated_metrics_list)
