# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:40:37 2019

@author: Cigar
"""

import os

PATH = "F:/for learn/asr_trial_data/"  # 原始数据存放地址
SAVE_PATH = os.path.join(PATH, "files")  # 处理后的数据存放地址
os.makedirs(SAVE_PATH, exist_ok=True)

MERGED_FILE = os.path.join(PATH, "merged_file.dat")  # 合并各个小txt的整合数据文件
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


