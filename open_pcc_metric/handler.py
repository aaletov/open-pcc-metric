import click


@click.command()
@click.option(
    "--ocloud",
    required=True,
    type=str,
    help="Original point cloud.",
)
@click.option(
    "--pcloud",
    required=True,
    type=str,
    help="Processed point cloud.",
)
@click.option(
    "--color",
    required=False,
    type=click.Choice(["rgb", "ycc"]),
    help="Report color distortions as well.",
)
@click.option(
    "--hausdorff",
    required=False,
    is_flag=True,
    help=" ".join((
        "Report hausdorff metric as well. If --point-to-plane is provided,",
        "then hausdorff point-to-plane would be reported too",
        )),
)
@click.option(
    "--point-to-plane",
    required=False,
    is_flag=True,
    help="Report point-to-plane distance as well.",
)
@click.option(
    "--csv",
    required=False,
    is_flag=True,
    help="Print output in csv format.",
)
def cli(
    ocloud: str,
    pcloud: str,
    color: str,
    hausdorff: bool,
    point_to_plane: bool,
    csv: bool,
) -> None:
    import open3d as o3d
    from .cloud_pair import CloudPair
    from .calculator import MetricCalculator
    from .options import CalculateOptions, transform_options

    ocloud_cloud, pcloud_cloud = map(o3d.io.read_point_cloud, (ocloud, pcloud))
    cloud_pair = CloudPair(ocloud_cloud, pcloud_cloud)
    calculator = MetricCalculator(cloud_pair)
    options = CalculateOptions(
        color=color,
        hausdorff=hausdorff,
        point_to_plane=point_to_plane,
    )
    metrics = transform_options(options)
    result = calculator.calculate(metrics).as_df()

    if csv:
        print(result.to_csv())
    else:
        print(result.to_string())
