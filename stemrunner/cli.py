import click
from pathlib import Path

from .pipeline import process_file
from .models import ModelManager

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument('files', nargs=-1, type=click.Path(exists=True))
@click.option('--gpu', type=int, default=None, help='GPU id to use; CPU if omitted.')
@click.option('--segment', type=int, default=None, help='Override segment size.')
@click.option('--outdir', type=click.Path(), default=None, help='Output directory.')
def main(files, gpu, segment, outdir):
    """CLI entry point for stemrunner."""
    if not files:
        click.echo('No input files provided.')
        return
    manager = ModelManager(gpu=gpu)
    for f in files:
        process_file(Path(f), manager, segment=segment, outdir=outdir)

if __name__ == '__main__':
    main()
