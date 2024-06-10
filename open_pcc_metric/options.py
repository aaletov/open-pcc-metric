import typing
from .metric import (
    AbstractMetric,
    MinSqrtDistance,
    MaxSqrtDistance,
    GeoMSE,
    GeoPSNR,
    ColorMSE,
    ColorPSNR,
    GeoHausdorffDistance,
    GeoHausdorffDistancePSNR,
    SymmetricMetric,
)


class CalculateOptions:
    color: typing.Optional[str]
    hausdorff: bool
    point_to_plane: bool

    def __init__(
        self,
        color: typing.Optional[str] = None,
        hausdorff: bool = False,
        point_to_plane: bool = False,
    ):
        self.color = color
        self.hausdorff = hausdorff
        self.point_to_plane = point_to_plane


def transform_options(
    options: 'CalculateOptions',
) -> typing.List['AbstractMetric']:
    metrics = [
        MinSqrtDistance(),
        MaxSqrtDistance(),
        GeoMSE(is_left=True, point_to_plane=False),
        GeoMSE(is_left=False, point_to_plane=False),
        SymmetricMetric(
            metrics=(
                GeoMSE(is_left=True, point_to_plane=False),
                GeoMSE(is_left=False, point_to_plane=False),
            ),
            is_proportional=False,
        ),
        GeoPSNR(is_left=True, point_to_plane=False),
        GeoPSNR(is_left=False, point_to_plane=False),
        SymmetricMetric(
            metrics=(
                GeoPSNR(is_left=True, point_to_plane=False),
                GeoPSNR(is_left=False, point_to_plane=False),
            ),
            is_proportional=True,
        ),
    ]

    if (
        # self.clouds[0].has_colors() and
        # self.clouds[1].has_colors() and
        (options.color is not None)
    ):
        metrics += [
            ColorMSE(is_left=True, color_scheme=options.color),
            ColorMSE(is_left=False, color_scheme=options.color),
            SymmetricMetric(
                metrics=(
                    ColorMSE(is_left=True, color_scheme=options.color),
                    ColorMSE(is_left=False, color_scheme=options.color),
                ),
                is_proportional=False,
            ),
            ColorPSNR(is_left=True, color_scheme=options.color),
            ColorPSNR(is_left=False, color_scheme=options.color),
            SymmetricMetric(
                metrics=(
                    ColorPSNR(is_left=True, color_scheme=options.color),
                    ColorPSNR(is_left=False, color_scheme=options.color),
                ),
                is_proportional=True,
            ),
        ]

    if options.point_to_plane:
        metrics += [
            GeoMSE(is_left=True, point_to_plane=True),
            GeoMSE(is_left=False, point_to_plane=True),
            SymmetricMetric(
                metrics=(
                    GeoMSE(is_left=True, point_to_plane=True),
                    GeoMSE(is_left=False, point_to_plane=True),
                ),
                is_proportional=False,
            ),
            GeoPSNR(is_left=True, point_to_plane=True),
            GeoPSNR(is_left=False, point_to_plane=True),
            SymmetricMetric(
                metrics=(
                    GeoPSNR(is_left=True, point_to_plane=True),
                    GeoPSNR(is_left=False, point_to_plane=True),
                ),
                is_proportional=True,
            ),
        ]

    if options.hausdorff:
        metrics += [
            GeoHausdorffDistance(is_left=True, point_to_plane=False),
            GeoHausdorffDistance(is_left=False, point_to_plane=False),
            SymmetricMetric(
                metrics=(
                    GeoHausdorffDistance(
                        is_left=True,
                        point_to_plane=False,
                    ),
                    GeoHausdorffDistance(
                        is_left=False,
                        point_to_plane=False,
                    ),
                ),
                is_proportional=False,
            ),
            GeoHausdorffDistancePSNR(is_left=True, point_to_plane=False),
            GeoHausdorffDistancePSNR(is_left=False, point_to_plane=False),
            SymmetricMetric(
                metrics=(
                    GeoHausdorffDistancePSNR(
                        is_left=True,
                        point_to_plane=False,
                    ),
                    GeoHausdorffDistancePSNR(
                        is_left=False,
                        point_to_plane=False,
                    ),
                ),
                is_proportional=True,
            ),
        ]

    if options.hausdorff and options.point_to_plane:
        metrics += [
            GeoHausdorffDistance(is_left=True, point_to_plane=True),
            GeoHausdorffDistance(is_left=False, point_to_plane=True),
            GeoHausdorffDistancePSNR(is_left=True, point_to_plane=True),
            GeoHausdorffDistancePSNR(is_left=False, point_to_plane=True),
            SymmetricMetric(
                metrics=(
                    GeoHausdorffDistance(
                        is_left=True,
                        point_to_plane=True,
                    ),
                    GeoHausdorffDistance(
                        is_left=False,
                        point_to_plane=True,
                    ),
                ),
                is_proportional=False,
            ),
            SymmetricMetric(
                metrics=(
                    GeoHausdorffDistancePSNR(
                        is_left=True,
                        point_to_plane=True,
                    ),
                    GeoHausdorffDistancePSNR(
                        is_left=False,
                        point_to_plane=True,
                    ),
                ),
                is_proportional=True,
            ),
        ]

    return metrics
