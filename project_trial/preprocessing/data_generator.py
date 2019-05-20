
import os
import re
import random
import numpy as np
import pandas as pd
from xpinyin import Pinyin
from project_trial.constant import MERGED_FILE, PURED_FILE, TRAIN_AUDIO_PATH, TEST_AUDIO_PATH, PATH, clear_path, MAX_TIME
from _utils.nlp.u_nlp import DBC2SBC
from scipy.io import wavfile
from tqdm import tqdm
from typing import Optional

random.seed(0)
tqdm.pandas(tqdm)  # can use tqdm_gui
PATTERN = re.compile("[\\u4e00-\\u9fa5_a-zA-Z0-9+\s]")
ENGLISH_PATTERN = re.compile("[a-zA-Z+]")
SPLIT_PATTERN = re.compile("[\,\.\?\!\:，。？！；;：]")
SPLITTER = "-"  # 生成拼音的分隔符
P = Pinyin()
TEST_SIZE = 0.2


def treat_content(content: str) -> str:
    content = re.sub("\s+", "", content)
    content = "".join([DBC2SBC(char) for char in content.strip()])  # 全角转半角
    splited_text = " ".join(SPLIT_PATTERN.split(content))
    return "".join(PATTERN.findall(splited_text)).lower()


def gen_pinyin(content: str) -> str:
    pinyin = P.get_pinyin(content, tone_marks="numbers", splitter=SPLITTER)
    return pinyin


def generate_data(source_file: str) -> pd.DataFrame:
    df = pd.read_table(source_file)
    df.fillna(value={"content": ""}, inplace=True)
    cond = (df["role"] == 0) & (df["noise"] == 0) & (df["content"].str.len() > 1) & (df["xmax"] - df["xmin"] <= MAX_TIME)
    user_df = df[cond].copy()
    user_df["pured_content"] = user_df["content"].progress_apply(treat_content)
    user_df = user_df[user_df["pured_content"].str.len() > 0]
    user_df["pinyin"] = user_df["pured_content"].progress_apply(gen_pinyin)
    user_df["id"] = range(len(user_df))  # 增加样本标号
    columns = ["id"] + list(df.columns) + ["pured_content", "pinyin"]
    return user_df[columns]


def fragment_audio(wav_file, xmin: float, xmax: float, write_file: str, sample_rate: Optional[int]=None):
    """将音频数据切割分片，并写入硬盘"""
    if isinstance(wav_file, str):
        sample_rate, wav_file = wavfile.read(wav_file)
    assert isinstance(wav_file, np.ndarray), "wav_file should be either the file path of a wav file or audio data"
    samples = wav_file[int(sample_rate * xmin): int(sample_rate * xmax) + 1]
    wavfile.write(filename=write_file, rate=sample_rate, data=samples)


def fragmented_grouply(df: pd.DataFrame):
    save_path = TRAIN_AUDIO_PATH if random.random() > TEST_SIZE else TEST_AUDIO_PATH
    head = df.iloc[0]
    audio_folder, audio_file = head["folder"], head["file"]
    sample_rate, wav_file = wavfile.read(os.path.join(PATH, audio_folder, audio_file + ".wav"))
    for _, row in df.iterrows():
        id_, xmin, xmax = row["id"], row["xmin"], row["xmax"]
        write_file = os.path.join(save_path, "%d.wav" % id_)
        fragment_audio(wav_file, xmin, xmax, write_file, sample_rate)


if __name__ == "__main__":
    dat = generate_data(MERGED_FILE)
    dat.to_csv(PURED_FILE, index=False, sep="\t")  # 保存已经处理好的可用数据集
    print("Saved valid data for training")
    for path in [TRAIN_AUDIO_PATH, TEST_AUDIO_PATH]:  # 清空目标文件夹
        clear_path(path)
    dat.groupby(["folder", "file"], as_index=False).progress_apply(fragmented_grouply)  # 生成与数据集相对应的语音文件
