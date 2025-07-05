from pathlib import Path
import sys
import torch
import torchaudio
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stemrunner.pipeline import process_file
from stemrunner.models import ModelManager


def test_process_file(tmp_path, monkeypatch):
    sample_rate = 44100
    tone = torch.zeros(1, int(sample_rate * 0.5))
    test_file = tmp_path / 'tone.wav'
    def fake_save(path, waveform, sample_rate, encoding=None):
        Path(path).write_bytes(b'')
    monkeypatch.setattr(torchaudio, 'save', fake_save)
    torchaudio.save(test_file, tone, sample_rate)
    monkeypatch.setattr(torchaudio, 'load', lambda p: (tone, sample_rate))
    manager = ModelManager(gpu=None)
    process_file(test_file, manager, outdir=tmp_path)
    out_dir = tmp_path / 'tone—stems'
    expected = [
        'tone—Vocals.wav',
        'tone—Instrumental.wav',
        'tone—Drums.wav',
        'tone—Bass.wav',
        'tone—Other.wav',
        'tone—Karaoke.wav',
        'tone—Guitar.wav',
    ]
    for name in expected:
        assert (out_dir / name).exists()


def test_ffmpeg_fallback(tmp_path, monkeypatch):
    sample_rate = 44100
    test_file = tmp_path / 'tone.m4a'

    def fake_save(path, waveform, sample_rate, encoding=None):
        Path(path).write_bytes(b'')

    monkeypatch.setattr(torchaudio, 'save', fake_save)

    def fail_load(p):
        raise RuntimeError('no backend')

    monkeypatch.setattr(torchaudio, 'load', fail_load)

    def fake_run(cmd, check, stdout=None, stderr=None):
        class R:
            def __init__(self):
                self.stdout = b''
        return R()

    monkeypatch.setattr(subprocess, 'run', fake_run)

    manager = ModelManager(gpu=None)
    process_file(test_file, manager, outdir=tmp_path)

    out_dir = tmp_path / 'tone—stems'
    for name in [
        'Vocals',
        'Instrumental',
        'Drums',
        'Bass',
        'Other',
        'Karaoke',
        'Guitar',
    ]:
        assert (out_dir / f'tone—{name}.wav').exists()
