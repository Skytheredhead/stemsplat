from pathlib import Path
import torchaudio

from .models import ModelManager

SEGMENT_STAGE_A = 352800
SEGMENT_STAGE_B = 4000
OVERLAP = 8


def process_file(path: Path, manager: ModelManager, segment: int | None = None, outdir: str | None = None):
    """Process a single audio file and save stems."""
    sample_rate = 44100
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    out_dir = Path(outdir or path.parent) / f"{path.stem}—stems"
    out_dir.mkdir(exist_ok=True)

    # Stage A
    vocals, instrumental = manager.split_vocals(waveform, segment or SEGMENT_STAGE_A, OVERLAP)
    torchaudio.save(out_dir / f"{path.stem}—Vocals.wav", vocals, sample_rate, encoding='PCM_S24LE')
    torchaudio.save(out_dir / f"{path.stem}—Instrumental.wav", instrumental, sample_rate, encoding='PCM_S24LE')

    # Stage B
    drums, bass, other = manager.split_instrumental(instrumental, SEGMENT_STAGE_B, OVERLAP)
    torchaudio.save(out_dir / f"{path.stem}—Drums.wav", drums, sample_rate, encoding='PCM_S24LE')
    torchaudio.save(out_dir / f"{path.stem}—Bass.wav", bass, sample_rate, encoding='PCM_S24LE')
    torchaudio.save(out_dir / f"{path.stem}—Other.wav", other, sample_rate, encoding='PCM_S24LE')
