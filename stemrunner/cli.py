import click
from main import cli_main


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument('files', nargs=-1, type=click.Path(exists=True))
@click.option('--stems', default='vocals', help='Comma-separated stems to extract when running locally.')
@click.option('--serve', is_flag=True, default=False, help='Start the FastAPI server instead of local processing.')
def main(files, stems, serve):
    """CLI entry point delegating to main.cli_main."""
    argv = []
    if serve:
        argv.append('--serve')
    argv.extend(['--stems', stems])
    argv.extend(list(files))
    cli_main(argv)


if __name__ == '__main__':
    main()
