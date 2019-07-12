
"""常量和常用函数"""

import os
import platform


system = platform.system().lower()
DATA_SOURCE_DIR = r"F:\for learn\data_source\Speech" if system == "windows"\
    else "/data/zhaochengming/data/data_source/Speech"

AM_LOG_DIR = os.path.join(os.path.dirname(__file__), "AM_Model_LOG")
os.makedirs(AM_LOG_DIR, exist_ok=True)
AM_MODEL_DIR = os.path.join(AM_LOG_DIR, "am_model.h5")
