from pathlib import Path
import sys
import wave
import struct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import types

# provide a dummy split.split.main before importing pipeline
dummy_split = types.ModuleType('split.split')
dummy_split.main = lambda args, progress_cb=None: None
sys.modules['split.split'] = dummy_split

from stemrunner.pipeline import process_file


def test_process_file_wav(tmp_path, monkeypatch):
    sample_rate = 44100
    n_frames = int(sample_rate * 0.5)
    test_file = tmp_path / 'tone.wav'
    with wave.open(str(test_file), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        silent = struct.pack('<' + 'h' * n_frames, *([0] * n_frames))
        wf.writeframes(silent)

    def fake_split(args, progress_cb=None):
        out = Path(args[args.index('--out') + 1])
        out.mkdir(exist_ok=True)
        (out / 'vocals.wav').write_bytes(b'')

    monkeypatch.setattr(sys.modules['stemrunner.pipeline'], 'split_main', fake_split)

    process_file(test_file, tmp_path / 'model.ckpt', outdir=tmp_path)
    out_dir = tmp_path / 'tone—stems'
    assert (out_dir / 'vocals.wav').exists()
