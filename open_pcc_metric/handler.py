import click

@click.command()
@click.option("--ocloud", required=True, type=str, help="Original point cloud.")
@click.option("--pcloud", required=True, type=str, help="Processed point cloud.")
def cli(ocloud: str, pcloud: str) -> None:
    from . import metric

    metrics = metric.calculate_from_files(ocloud_file=ocloud, pcloud_file=pcloud)
    for k, v in metrics.items():
        print("{key}: {value}".format(key=k, value=v))