import numpy as np

float32 = np.float32
int16 = np.int16


def zeros(*shape, dtype=float32):
    return np.zeros(shape, dtype=dtype)

def frombuffer(buf, dtype):
    return np.frombuffer(buf, dtype=dtype)


def zeros_like(arr, dtype=None):
    return np.zeros_like(arr, dtype=dtype if dtype is not None else arr.dtype)

class Device:
    def __init__(self, type):
        self.type = type

class backends:
    class mps:
        @staticmethod
        def is_available():
            return False

def device(type):
    return Device(type)
