""" 操作原始数据集，生成可用的结构化标注文件 """

import os
import sys
import re
import glob
import tqdm
import platform
import argparse
from typing import Optional

system = platform.system().lower()
sys.path.append("F:/Code projects/Python/SpeechRecognition" if system == "windows"
                else "/data/zhaochengming/projects/SpeechRecognition")

from deep_speech_by_audier.constant import DATA_SOURCE_DIR

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


def _get_partition_files(dirname: str):
    return set([os.path.splitext(file)[0] for file in os.listdir(dirname) if file.endswith(".wav")])


def _transform_thchs30(source_dir: str, output_file: str, correction_file: Optional[str]=None):
    """
    解析 THCHS30 开源数据集，生成结构化标注信息
    :param source_dir: 下载好的 data_thchs30文件夹地址
    :param output_file: 输出的结构化文本文件
    """
    correction = _correction_dict(correction_file) if correction_file else {}
    data_dir, train_dir, test_dir, dev_dir = [os.path.join(source_dir, x) for x in ["data", "train", "test", "dev"]]
    train_sets, test_sets, dev_sets = [_get_partition_files(x) for x in [train_dir, test_dir, dev_dir]]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fw:
        for file in tqdm.tqdm(glob.glob(os.path.join(data_dir, "*.trn"))):
            wav_file = file.strip(".trn")
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
            fw.write("\t".join([wav_file, content, pinyin, partition]) + "\n")
    print("Transformed Done! Successfully written into %s" % output_file)


def _transform_aishell(source_dir: str, output_file: str, correction_file: Optional[str]=None):
    pass


def transform(args):
    if args.thchs30:
        source_dir, *files = args.thchs30
        if files:
            correction_file, *output_file = files
            if not output_file:
                output_file = os.path.join(DATA_SOURCE_DIR, "labeled_data.txt")
        else:
            correction_file = os.path.join(source_dir, "pinyin_correction.txt")
            output_file = os.path.join(DATA_SOURCE_DIR, "labeled_data.txt")
        _transform_thchs30(source_dir, output_file, correction_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thchs30", type=str, help="where to load thchs source and where to sink result", nargs="*")
    parser.add_argument("--aishell", type=str, help="where to load aishell source and where to sink result", nargs="*")

    args = parser.parse_args()
    transform(args)
    # python arrangement.py --thchs30 "F:/for learn/data_source/Speech/THCHS30/data_thchs30"
