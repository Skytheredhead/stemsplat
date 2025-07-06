import wave
import numpy as np

__all__ = ['save', 'load', 'functional']

def save(path, waveform, sample_rate, encoding=None):
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    data = (arr.T * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data.tobytes())

def load(path):
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        frames = w.readframes(w.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).reshape(-1, nch).T.astype(np.float32) / 32768.0
    return data, sr

class functional:
    @staticmethod
    def resample(wav, sr, new_sr):
        return wav
