"""Read audio file"""

import numpy as np
import soundfile as sf
from scipy.io import wavfile
from typing import Tuple


def read_audio(file: str) -> Tuple[int, np.ndarray]:
    file = file.lower()
    if file.endswith(".wav"):
        fs, samples = wavfile.read(file)
    else:
        samples, fs = sf.read(file)
    return fs, samples
