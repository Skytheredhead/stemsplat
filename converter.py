from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import List

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

DEFAULT_INPUT_DIR = Path.cwd() / "uploads"
DEFAULT_OUTPUT_DIR = Path.cwd() / "uploads_converted"
SUPPORTED_EXTENSIONS: List[str] = [
    ".wav",
    ".mp3",
    ".aac",
    ".flac",
    ".ogg",
    ".alac",
    ".opus",
    ".m4a",
]

FFMPEG_CMD = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel",
    "error",  # show only errors; change to "info" for verbose output
]

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def check_ffmpeg_available() -> None:
    """Verify FFmpeg is installed and available in PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit("❌  FFmpeg not found. Install it and ensure it’s on your PATH.")


def convert_to_wav(src: Path, dst: Path) -> None:
    """Convert a single audio file to 44.1 kHz 16‑bit stereo WAV using FFmpeg."""

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        *FFMPEG_CMD,
        "-y",  # overwrite existing files without prompting
        "-i",
        str(src),
        "-ac",
        "2",  # stereo
        "-ar",
        "44100",  # sample rate 44.1 kHz
        "-sample_fmt",
        "s16",  # 16‑bit PCM
        str(dst),
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅  Converted: {src.name} → {dst.relative_to(dst.parents[1])}")
    except subprocess.CalledProcessError as exc:
        print(f"⚠️  Failed to convert {src.name}: {exc}")


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def main(input_dir: Path = DEFAULT_INPUT_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    check_ffmpeg_available()

    if not input_dir.exists() or not input_dir.is_dir():
        sys.exit(f"❌  Input directory not found: {input_dir}")

    if output_dir.exists():
        print(f"ℹ️  Output directory exists and will be reused: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [p for p in input_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not audio_files:
        print("ℹ️  No supported audio files found. Nothing to do.")
        return

    print(f"🔄  Converting {len(audio_files)} file(s)…\n")

    for src_path in audio_files:
        relative_subpath = src_path.relative_to(input_dir).with_suffix(".wav")
        dst_path = output_dir / relative_subpath
        convert_to_wav(src_path, dst_path)

    print("\n🎉  Done! Converted files are in:", output_dir)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 3:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        print("Usage: python convert_audio_pipeline.py [input_dir output_dir]")
        sys.exit(1)
