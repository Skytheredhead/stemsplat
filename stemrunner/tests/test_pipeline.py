from pathlib import Path
import torchaudio

from stemrunner.pipeline import process_file
from stemrunner.models import ModelManager


def test_process_file(tmp_path):
    sample_rate = 44100
    tone = torchaudio.functional.generate_tone(frequency=440, sample_rate=sample_rate, duration=0.5)
    test_file = tmp_path / 'tone.wav'
    torchaudio.save(test_file, tone, sample_rate)
    manager = ModelManager(gpu=None)
    process_file(test_file, manager, outdir=tmp_path)
    out_dir = tmp_path / 'tone—stems'
    assert (out_dir / 'tone—Vocals.wav').exists()
    assert (out_dir / 'tone—Drums.wav').exists()
