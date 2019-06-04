# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:03:15 2019

@author: Cigar
"""
import numpy as np
import os
import pandas as pd
import re
import xpinyin
from pypinyin import pinyin, Style, slug, pinyin_dict
from pypinyin.core import to_fixed
from collections import defaultdict


"""
结构化标注数据的拼音标注异常检测
输出修正文件 pinyin_correction.txt到相应数据集目录下
"""


path = r"F:\for learn\data_source\Speech\THCHS30\data_thchs30"

def deal_with_no_tone(p):
    if p[-1] not in ("1", "2", "3", "4", "5"):
        p += "5"
    return p


pattern = re.compile("[\u4e00-\u9fa5]")
P = xpinyin.Pinyin()
data = pd.read_csv(os.path.join(path, "thchs30.txt"), sep="\t", header=None, names=["src", "content", "pinyin", "partition"])
data["mapping"] = data.apply(lambda row: list(zip(row["content"], row["pinyin"].split())), axis=1)

d = defaultdict(list)
for _, row in data.iterrows():
    for w, pny in zip(list(row["content"]), row["pinyin"].split()):
        d[w].append(pny)

d = {k: list(set(v)) for k, v in d.items()}
#all_d = {chr(int(k, 16)): v.strip().lower().split() for k, v in P.dict.items()}
all_d = {chr(k): [deal_with_no_tone(to_fixed(x, strict=True, style=Style.TONE3)) for x in v.split(",")] for k, v in pinyin_dict.pinyin_dict.items()}
pinyin = sorted(list(set(" ".join(data["pinyin"]).split())))
#all_pinyin = sorted(list(set(" ".join(P.dict.values()).replace("\n", "").lower().split())))
all_pinyin = sorted(list(set([to_fixed(py, strict=True, style=Style.TONE3) for py in ",".join(pinyin_dict.pinyin_dict.values()).split(",")])))

log = {}
for word, pnys in d.items():
    target_pnys = all_d[word]
    diff = [pny for pny in pnys if pny not in target_pnys]
    if diff:
        log[word] = (diff, target_pnys)

with open(os.path.join(path, "pinyin_correction.txt"), "w", encoding="utf-8") as f:
    rest = {}
    for word, (diff_pnys, target_pnys) in log.items():
        treated = False
        if len(target_pnys) == 1:
            treated = True
            for pny in diff_pnys:
                f.write("%s\t%s\t%s\n" % (word, pny, target_pnys[0]))
        else:
            if len(diff_pnys) == 1:
                pny = diff_pnys[0]
                if pny.endswith("2") and pny.replace("2", "3") in target_pnys:
                    treated = True
                    f.write("%s\t%s\t%s\n" % (word, pny, pny.replace("2", "3")))
                elif pny.endswith("5"):
                    if pny.replace("5", "4") in target_pnys:
                        treated = True
                        f.write("%s\t%s\t%s\n" % (word, pny, pny.replace("5", "4")))
                    elif pny.replace("5", "1") in target_pnys:
                        treated = True
                        f.write("%s\t%s\t%s\n" % (word, pny, pny.replace("5", "1")))
        if not treated:
            rest[word] = (diff_pnys, target_pnys)


####################
""" 探查及修复"""

df = data[data["mapping"].apply(lambda x: ("弟", "de5") in x)]

for src in df["src"]:
    with open(src, "r+", encoding="utf-8") as f:
        content = [line.replace("已", "己") for line in f.readlines()]
        f.seek(0)
        f.truncate()
        f.writelines(content)


def _correction_dict(correction_file: str):
    d = {}
    with open(correction_file, "r", encoding="utf-8") as f:
        i = 0
        for line in f:
            line = line.strip()
            if line:
                word, old_py, new_py = line.split("\t")
                d[(word, old_py)] = new_py
            i += 1
    return d
