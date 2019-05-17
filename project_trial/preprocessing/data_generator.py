
import os
import re
import pandas as pd
from xpinyin import Pinyin
from project_trial.constant import MERGED_FILE
from _utils.nlp.u_nlp import DBC2SBC
from tqdm import tqdm

tqdm.pandas(tqdm)  # can use tqdm_gui
# MERGED_FILE = "F:/for learn/asr_trial_data/merged_file.dat"
PATTERN = re.compile("[\\u4e00-\\u9fa5_a-zA-Z0-9+\s]")
ENGLISH_PATTERN = re.compile("[a-zA-Z+]")
SPLIT_PATTERN = re.compile("\,|\.|\?|\!|，|。|？|！|；|;|\:|：")
P = Pinyin()


def treat_content(content):
    content = re.sub("\s+", "", content)
    content = "".join([DBC2SBC(char) for char in content.strip()])
    splited_text = " ".join(SPLIT_PATTERN.split(content))
    return "".join(PATTERN.findall(splited_text)).lower()

def gen_pinyin_label(content):
    pinyin = P.get_pinyin(content,tone_marks="numbers", splitter="-")
    return pinyin

df = pd.read_table(MERGED_FILE)
df.fillna(value={"content": ""}, inplace=True)
cond = (df["role"] == 0) & (df["noise"] == 0) & (df["content"].str.len() > 1)
user_df = df[cond].copy()
user_df["pured_content"] = user_df["content"].progress_apply(treat_content)
user_df = user_df[user_df["pured_content"].str.len() > 0]
user_df["pinyin"] = user_df["pured_content"].progress_apply(gen_pinyin_label)
