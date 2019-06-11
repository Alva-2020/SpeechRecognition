
import os
import glob
import argparse
from scipy.io import wavfile


def total_wav_length(dir: str) -> float:
    total_seconds = 0
    for file in glob.iglob(os.path.join(dir, "**/*.wav"), recursive=True):
        fs, audio = wavfile.read(file)
        seconds = len(audio) / fs
        total_seconds += seconds
    print("%s: %.2f hrs" % (dir, total_seconds / 3600))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="the target dir to summary", nargs="*")
    args = parser.parse_args()
    if args.dir:
        for dir in args.dir:
            total_wav_length(dir)
