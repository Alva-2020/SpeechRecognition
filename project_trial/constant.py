# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:40:37 2019

@author: Cigar
"""

import os
from xpinyin import Pinyin
from typing import Dict

PATH = "F:/for learn/asr_trial_data/"  # 原始数据存放地址
SAVE_PATH = os.path.join(PATH, "files")  # 处理后的数据存放地址
FRAGMENTED_AUDIO_PATH = os.path.join(PATH, "fragmented_audios")  # 存放切片语音数据的位置，与文本对应
TRAIN_AUDIO_PATH = os.path.join(FRAGMENTED_AUDIO_PATH, "train")  # 训练数据存放位置
TEST_AUDIO_PATH = os.path.join(FRAGMENTED_AUDIO_PATH, "test")  # 训练测试数据的存放位置

for _path in [SAVE_PATH, FRAGMENTED_AUDIO_PATH, TRAIN_AUDIO_PATH, TEST_AUDIO_PATH]:
    os.makedirs(_path, exist_ok=True)

MERGED_FILE = os.path.join(PATH, "merged_file.dat")  # 合并各个小txt的整合数据文件
PURED_FILE = os.path.join(PATH, "pured_file.dat")  # 用于训练测试的可用数据集文件
"""
文件列名枚举值说明:
sex性别:
    - 0: 女性
    - 1: 男性
    - -1: 不确定
role身份：
    - 0: 用户
    - 1: 客服
    - -1: 无法识别
noise噪声:
    - 0: 非噪声
    - 1: 噪声
    - -1: 无法识别
"""
MAX_TIME = 5  # 最大时间长度
SPACE_INDEX = 0  # ctc loss的space
SEP_INDEX = 1  # 真实的断句符
N_CLASSES = 1468  # pinyin(1430) + numbers(10) + letters(26) + <SPACE> + <SEP>


def _get_total_labels() -> Dict[str, int]:
    """生成全部 字符到 label的 词典"""
    all_pinyin = list(set(" ".join(Pinyin().dict.values()).replace("\n", "").lower().split()))  # 1430
    all_numbers = [str(n) for n in range(10)]
    all_letters = [chr(x) for x in range(ord("a"), ord("z") + 1)]
    res = {x: i + 2 for i, x in enumerate(all_numbers + all_letters + all_pinyin)}  # 10 + 26 + 1430
    res[" "] = SEP_INDEX  # 增加空格
    return res


LABEL_MAP = _get_total_labels()  # {字符: label_index}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}  # 反转mapping {label_index: 字符}


def clear_path(path: str):
    # os.removedirs(path)
    # os.makedirs(path)
    for x in os.listdir(path):
        os.remove(os.path.join(path, x))
