from __future__ import annotations

from pathlib import Path

from main import convert_directory


def main(input_dir: Path | None = None, output_dir: Path | None = None) -> None:
    in_dir = input_dir or (Path.cwd() / "uploads")
    out_dir = output_dir or (Path.cwd() / "uploads_converted")
    converted = convert_directory(in_dir, out_dir)
    for path in converted:
        print(path)


if __name__ == "__main__":
    main()
