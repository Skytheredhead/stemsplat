import click
from pathlib import Path

from .pipeline import process_file
from .models import ModelManager

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument('files', nargs=-1, type=click.Path(exists=True))
@click.option('--gpu', type=int, default=None, help='GPU id to use; CPU if omitted.')
<<<<<<< HEAD
@click.option('--outdir', type=click.Path(), default=None, help='Output directory.')
def main(files, gpu, outdir):
=======
@click.option('--segment', type=int, default=None, help='Override segment size.')
@click.option('--outdir', type=click.Path(), default=None, help='Output directory.')
def main(files, gpu, segment, outdir):
>>>>>>> 81855b9 (VOCALS WORKINGgit add .)
    """CLI entry point for stemrunner."""
    if not files:
        click.echo('No input files provided.')
        return
    manager = ModelManager(gpu=gpu)
<<<<<<< HEAD
    ckpt = getattr(manager.vocals, 'path', None)
    if ckpt is None:
        click.echo('Checkpoint not found.')
        return
    for f in files:
        process_file(Path(f), ckpt, outdir=outdir)
=======
    for f in files:
        process_file(Path(f), manager, segment=segment, outdir=outdir)
>>>>>>> 81855b9 (VOCALS WORKINGgit add .)

if __name__ == '__main__':
    main()
