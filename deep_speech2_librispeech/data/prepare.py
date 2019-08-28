"""Process on LibriSpeech Data downloaded."""


import os
import glob
import argparse
import pandas as pd
import soundfile as sf
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Tuple


tqdm.pandas()
Data = namedtuple("Data", ["src", "duration", "transcript"])


def get_audio_duration(audio_file: str) -> float:
    data, sample_rate = sf.read(audio_file)
    return len(data) / sample_rate


def convert_audio_and_split_transcript(trans_file: str):
    if not os.path.exists(trans_file):
        raise IOError(f"File '{trans_file}' not exists.")
    dirname = os.path.dirname(trans_file)
    collections = []
    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seqid, transcript = line.split(" ", maxsplit=1)
            transcript = transcript.lower()
            audio_file = os.path.join(dirname, seqid + ".flac")
            duration = get_audio_duration(audio_file)
            collections.append(Data(src=audio_file, duration=duration, transcript=transcript))
    return collections


def get_info(src: str, start: str) -> Tuple[str, str]:
    rel_path: str = os.path.relpath(src, start).replace("\\", "/")
    info_pair = rel_path.split("/", maxsplit=1)[0]
    partition, tag = info_pair.split("-")
    return partition, tag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, help="The LibriSpeech data path.", required=True)
    parser.add_argument("--output_path", type=str, help="The path where output file written in.", required=True)
    args = parser.parse_args()

    read_path = args.source_path
    output_path = args.output_path

    transcript_files = glob.glob(os.path.join(read_path, "**/*.trans.txt"), recursive=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        total_collections = list(pool.map(convert_audio_and_split_transcript, tqdm(transcript_files)))

    df = pd.DataFrame(data=[data for collection in total_collections for data in collection])
    df["partition"], df["tag"] = zip(*df["src"].progress_apply(lambda src: get_info(src, read_path)))
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print("Processing Successfully!")









