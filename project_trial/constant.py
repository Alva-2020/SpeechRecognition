# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:40:37 2019

@author: Cigar
"""

import os

PATH = "F:/for learn/asr_trial_data/"  # 原始数据存放地址
SAVE_PATH = os.path.join(PATH, "files")  # 处理后的数据存放地址
os.makedirs(SAVE_PATH, exist_ok=True)