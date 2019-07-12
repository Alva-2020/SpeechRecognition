
""" 操作原始数据集，生成可用的结构化标注文件 """

import os
import re
import glob
import tqdm
import argparse
from scipy.io import wavfile
from collections import namedtuple
from pypinyin import slug, Style
from _utils.nlp.u_nlp import DBC2SBC
from typing import Optional, Dict, Tuple
from deep_speech2_baidu.constant import DATA_SOURCE_DIR

Data = namedtuple("Data", ["src", "duration", "content", "pinyin", "partition", "data_source"])
HAN_PATTERN = re.compile('[\u4e00-\u9fa5]')  # 汉字


def _correction_dict(correction_file: str):
    d = {}
    with open(correction_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                word, old_py, new_py = line.split("\t")
                d[(word, old_py)] = new_py
    return d


def _get_duration(wav_file: str) -> float:
    sample_rate, audio_data = wavfile.read(wav_file)
    return float(len(audio_data) / sample_rate)


def _get_partition_files_thch30(dirname: str):
    return set([os.path.splitext(file)[0] for file in os.listdir(dirname) if file.endswith(".wav")])


def _transform_thchs30(source_dir: str, correction_file: Optional[str]=None):
    """
    解析 THCHS30 开源数据集，生成结构化标注信息
    :param source_dir: 下载好的 data_thchs30文件夹地址
    :param correction_file: 纠错表，线下整理生成
    """
    correction = _correction_dict(correction_file) if correction_file else {}
    data_dir, train_dir, test_dir, dev_dir = [os.path.join(source_dir, x) for x in ["data", "train", "test", "dev"]]
    train_sets, test_sets, dev_sets = [_get_partition_files_thch30(x) for x in [train_dir, test_dir, dev_dir]]

    for file in tqdm.tqdm(glob.glob(os.path.join(data_dir, "*.trn"))):
        wav_file = file.strip(".trn")
        duration = _get_duration(wav_file)
        wav_name = os.path.splitext(os.path.basename(wav_file))[0]
        with open(file, "r", encoding="utf-8") as fr:
            content, pinyin, *_ = fr.readlines()
        content = HAN_PATTERN.findall(content)  # 只取汉字部分
        pinyin = pinyin.strip().split()

        if correction:
            corrected_pinyin = pinyin.copy()
            for i, (word, old_py) in enumerate(zip(content, pinyin)):
                if (word, old_py) in correction:
                    corrected_pinyin[i] = correction[(word, old_py)]
            pinyin = corrected_pinyin

        content = "".join(content)
        pinyin = "-".join(pinyin)
        partition = ""
        if wav_name in train_sets:
            partition = "train"
        elif wav_name in test_sets:
            partition = "test"
        elif wav_name in dev_sets:
            partition = "dev"

        yield Data(src=wav_file, duration=duration, content=content,
                   pinyin=pinyin, partition=partition, data_source="thchs30")


def _fix_no_tone(pny: str):
    if pny[-1] not in {"1", "2", "3", "4"}:  # 轻声处理
        if pny.islower():
            pny += "5"
    return pny


def _load_transcript_aishell(transcript_file: str) -> Dict[str, Tuple[str, str]]:
    res = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                wav_file, content = line.split(" ", maxsplit=1)
                content = "".join([DBC2SBC(char) for char in content]).upper().replace(" ", "")
                pinyin = slug(content, style=Style.TONE3, separator=" ")
                pinyin = "-".join([_fix_no_tone(pny) for pny in pinyin.split()])
                res[wav_file] = (content, pinyin)
    return res


def _transform_aishell(source_dir: str, correction_file: str):
    """
    解析 aishell 开源数据集，生成结构化标注信息
    :param source_dir: 下载好的 data_aishell 文件夹地址
    :param correction_file: 修正文档（暂时无用）
    """
    transcript_file = glob.glob(os.path.join(source_dir, "transcript/*.txt"))[0]
    wav_dir = os.path.join(source_dir, "wav")
    file_content_mapping = _load_transcript_aishell(transcript_file)
    for partition in ["train", "test", "dev"]:
        for wav_file in tqdm.tqdm(glob.glob(os.path.join(wav_dir, partition, "*/*.wav"))):
            duration = _get_duration(wav_file)
            filename, _ = os.path.splitext(os.path.basename(wav_file))
            if filename in file_content_mapping:
                content, pinyin = file_content_mapping[filename]
                yield Data(src=wav_file, duration=duration, content=content,
                           pinyin=pinyin, partition=partition, data_source="aishell")


def transform(args):
    output_file = args.output if args.output else os.path.join(DATA_SOURCE_DIR, "labeled_data.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    stream = []

    if args.thchs30:
        source_dir, *correction_file = args.thchs30
        correction_file = correction_file[0] if correction_file else os.path.join(source_dir, "pinyin_correction.txt")
        stream.append(_transform_thchs30(source_dir, correction_file))

    if args.aishell:
        source_dir, *correction_file = args.aishell
        correction_file = correction_file[0] if correction_file else None
        stream.append(_transform_aishell(source_dir, correction_file))

    with open(output_file, "w", encoding="utf-8") as fw:
        for g in stream:
            for wav_file, duration, content, pinyin, partition, data_source in g:
                fw.write("\t".join([wav_file, duration, content, pinyin, partition, data_source]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thchs30", type=str, help="where to load thchs source and correction file", nargs="*")
    parser.add_argument("--aishell", type=str, help="where to load aishell source and correction file", nargs="*")
    parser.add_argument("--output", type=str, default=None, help="where to save result")

    args = parser.parse_args()
    transform(args)

    # python arrangement.py
    #   --thchs30 "/data/zhaochengming/data/data_source/Speech/THCHS30/data_thchs30"
    #   --aishell "/data/zhaochengming/data/data_source/Speech/aishell/data_aishell"
    #   --output "/data/zhaochengming/data/data_source/Speech/labeled_data.txt"
