

import os
import platform
from pypinyin import Style, pinyin_dict
from pypinyin.core import to_fixed
from typing import Dict

system = platform.system().lower()
DATA_SOURCE_DIR = r"F:\for learn\data_source\Speech" if system == "windows"\
    else "/data/zhaochengming/data/data_source/Speech"

AM_LOG_DIR = os.path.join(os.path.dirname(__file__), "AM_Model_LOG")
os.makedirs(AM_LOG_DIR, exist_ok=True)
AM_MODEL_DIR = os.path.join(AM_LOG_DIR, "am_model.h5")


def _fix_pny(pny):
    pny = to_fixed(pny, strict=True, style=Style.TONE3)
    if pny[-1] not in ("1", "2", "3", "4"):  # 轻声处理
        pny += "5"
    return pny


def make_vocab() -> Dict[str, int]:
    pinyin_values = ",".join(pinyin_dict.pinyin_dict.values()).split(",")
    all_pinyin = set([_fix_pny(pny) for pny in pinyin_values])
    all_pinyin = sorted(list(all_pinyin))
    constant_values = ["_"]
    all_numbers = [str(i) for i in range(10)]
    all_letters = [chr(x) for x in range(ord("a"), ord("z") + 1)]
    all_labels = constant_values + all_numbers + all_letters + all_pinyin
    return {pny: i for i, pny in enumerate(all_labels)}  # 1585




