import click

@click.command()
@click.option("--ocloud", required=True, type=str, help="Original point cloud.")
@click.option("--pcloud", required=True, type=str, help="Processed point cloud.")
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
def cli(ocloud: str, pcloud: str, color: str, hausdorff: bool, point_to_plane: bool) -> None:
    from . import metric

    options = metric.CalculateOptions(
        color=color,
        hausdorff=hausdorff,
        point_to_plane=point_to_plane,
    )

    result = metric.calculate_from_files(
        ocloud_file=ocloud,
        pcloud_file=pcloud,
        calculate_options=options,
    )

    print(result.to_string())
