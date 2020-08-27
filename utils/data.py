"""Base data loader of known source"""

import os
import re
import abc
import glob
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
from pypinyin import lazy_pinyin, Style
from evan_utils.nlp import DBC2SBC, fix_no_tone
from .audio import read_audio
from typing import Tuple, Set, Iterator, Dict, Iterable, List


HAN_PATTERN = re.compile('[\u4e00-\u9fa5]')  # 汉字
Data = namedtuple("Data", ["src", "duration", "content", "transcript", "partition", "tag"])


def stream_write(output_file: str, stream: Iterable[Iterator[Data]]):
    """Write streaming data into disk file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    data = pd.DataFrame(data=[row for it in stream for row in it])
    data.to_csv(output_file, sep="\t", index=False)


class BaseLoader(metaclass=abc.ABCMeta):
    """Base structure of loading data."""

    def __init__(self, source_dir: str):
        print(f"transform data in {source_dir}")
        if not os.path.exists(source_dir):
            raise IOError(f"{source_dir} not found!")
        _valid, msg = self._validate_dir(source_dir)
        if not _valid:
            raise IOError(f"{source_dir} is not valid. {msg}")
        self.source_dir = source_dir
        self.data_it = self.transform(self.source_dir)

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @staticmethod
    def get_duration(audio_file: str):
        """Get the duration of audio in seconds"""
        fs, samples = read_audio(audio_file)
        return len(samples) / fs

    @staticmethod
    @abc.abstractmethod
    def _validate_dir(source_dir: str) -> Tuple[bool, str]:
        pass

    @abc.abstractmethod
    def transform(self, source_dir: str) -> Iterator[Data]:
        pass


class THCHS30Loader(BaseLoader):
    """Method to load THCHS30 data."""

    def __init__(self, source_dir: str):
        super(THCHS30Loader, self).__init__(source_dir)

    @property
    def name(self):
        return "thchs30"

    @staticmethod
    def _validate_dir(source_dir: str) -> Tuple[bool, str]:
        if "data" not in os.listdir(source_dir):
            return False, f"`data` not under {source_dir}."
        return True, ""

    @staticmethod
    def _get_partition_files(dirname: str) -> Set[str]:
        """Get file names from given dirname"""
        return set([os.path.splitext(file)[0] for file in os.listdir(dirname) if file.endswith(".wav")])

    @staticmethod
    def read_trn(trn_file: str) -> Tuple[str, str]:
        with open(trn_file, "r", encoding="utf-8") as fr:
            content, pinyin, *_ = fr.readlines()
        content = HAN_PATTERN.findall(content)  # 只取汉字部分
        content = "".join(content)
        pinyin = lazy_pinyin(content, style=Style.TONE3, errors="ignore", strict=True)
        pinyin = "-".join([fix_no_tone(pny) for pny in pinyin])
        return content, pinyin

    def transform(self, source_dir: str) -> Iterator[Data]:
        data_dir, train_dir, test_dir, dev_dir =\
            [os.path.join(source_dir, folder) for folder in ["data", "train", "test", "dev"]]
        train_sets, test_sets, dev_sets =\
            [self._get_partition_files(dirname) for dirname in [train_dir, test_dir, dev_dir]]

        for file in tqdm(glob.glob(os.path.join(data_dir, "*.trn")), desc=f"Loading {self.name} data"):
            wav_file = file.strip(".trn")
            wav_name = os.path.splitext(os.path.basename(wav_file))[0]

            if wav_name in train_sets:
                partition = "train"
            elif wav_name in test_sets:
                partition = "test"
            elif wav_name in dev_sets:
                partition = "dev"
            else:
                partition = ""

            duration = self.get_duration(wav_file)
            content, transcript = self.read_trn(file)
            yield Data(
                src=wav_file, duration=duration, content=content,
                transcript=transcript, partition=partition, tag=self.name)


class AiShellLoader(BaseLoader):
    """Method to load AiShell v1 data."""

    def __init__(self, source_dir: str):
        super(AiShellLoader, self).__init__(source_dir)

    @property
    def name(self):
        return "aishell"

    @staticmethod
    def _validate_dir(source_dir: str) -> Tuple[bool, str]:
        if "wav" not in os.listdir(source_dir):
            return False, f"`wav` not under {source_dir}."
        return True, ""

    @staticmethod
    def load_transcripts(file: str) -> Dict[str, Tuple[str, str]]:
        res = {}
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                wav_file, content = line.split(" ", maxsplit=1)
                content = "".join([DBC2SBC(char) for char in content.upper().replace(" ", "")])
                content = "".join(HAN_PATTERN.findall(content))
                pinyin = lazy_pinyin(content, style=Style.TONE3, errors="ignore", strict=True)
                pinyin = "-".join([fix_no_tone(pny) for pny in pinyin])
                res[wav_file] = (content, pinyin)
        return res

    def transform(self, source_dir: str) -> Iterator[Data]:
        transcript_file = glob.glob(os.path.join(source_dir, "transcript/*.txt"))[0]
        wav_dir = os.path.join(source_dir, "wav")
        file_content_mapping = self.load_transcripts(transcript_file)
        for partition in ["train", "test", "dev"]:
            partition_path = os.path.join(wav_dir, partition)
            for wav_file in tqdm(glob.glob(os.path.join(partition_path, "*/*.wav")),
                                 desc=f"Loading {self.name} data {partition}"):
                filename, _ = os.path.splitext(os.path.basename(wav_file))
                if filename in file_content_mapping:
                    duration = self.get_duration(wav_file)
                    content, transcript = file_content_mapping[filename]
                    yield Data(
                        src=wav_file, duration=duration, content=content,
                        transcript=transcript, partition=partition, tag=self.name)


class LibriSpeechLoader(BaseLoader):
    def __init__(self, source_dir: str):
        super(LibriSpeechLoader, self).__init__(source_dir)

    @property
    def name(self):
        return "libri-speech"

    @staticmethod
    def _validate_dir(source_dir: str) -> Tuple[bool, str]:
        if "train-clean-100" not in os.listdir(source_dir):
            return False, f"`train-clean-100` not under {source_dir}."
        return True, ""

    @staticmethod
    def get_rel_info(src: str, start: str) -> Tuple[str, str]:
        rel_path: str = os.path.relpath(src, start).replace("\\", "/")
        info_pair = rel_path.split("/", maxsplit=1)[0]
        partition, tag, *_ = info_pair.split("-", maxsplit=2)
        return partition, tag

    @staticmethod
    def read_trans(trans_file: str) -> List[Tuple[str, str]]:
        dirname = os.path.dirname(trans_file)
        res = []
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                seq_id, transcript = line.split(" ", maxsplit=1)
                audio_file = os.path.join(dirname, seq_id + ".flac")
                transcript = transcript.lower()
                res.append((audio_file, transcript))
        return res

    def transform(self, source_dir: str) -> Iterator[Data]:
        for trans_file in tqdm(glob.glob(os.path.join(source_dir, "**/*.trans.txt"), recursive=True)):
            partition, tag = self.get_rel_info(trans_file, source_dir)
            tag = self.name + tag
            for audio_file, transcript in self.read_trans(trans_file):
                duration = self.get_duration(audio_file)
                yield Data(src=audio_file, duration=duration, content=transcript,
                           transcript=transcript, partition=partition, tag=tag)
