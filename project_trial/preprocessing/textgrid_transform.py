
"""解析 TextGrid，生成结构化文本，并存入SAVE_PATH中"""

import os
import glob
import tqdm
import textgrid
from collections import defaultdict
from project_trial import constant
from typing import Dict, List, Tuple

data_path = constant.PATH
save_path = constant.SAVE_PATH


def get_folder_file_name(file: str) -> (str, str):
    """解析文件路径，获取上级路径名称和文件名称（不含后缀名）"""
    full_path, full_filename = os.path.split(file)
    filename, suffix = os.path.splitext(full_filename)
    direct_dir = os.path.basename(full_path)
    return direct_dir, filename


def transform_textgrid(file: str) -> (Dict[Tuple[str, str], List[str]], List[str]):
    """抽取 textgrid文件中的数据信息"""
    res = defaultdict(list)
    grid = textgrid.TextGrid.load(file)
    name_list = []
    for tier in grid:
        name = tier.nameid
        name_list.append(name)
        for xmin, xmax, value in tier.simple_transcript:
            res[(xmin, xmax)].append(value)
    return dict(res), name_list


def write(data: Dict[Tuple[str, str], List[str]], name_list: List[str], target_dir: str):
    """
    将抽取的信息写入文本文件
    :param data: 抽取的信息
    :param name_list: 属性列表，用于写入第一行
    :param target_dir: 写入文件
    :return: unit
    """
    first_line = ["xmin", "xmax"] + name_list
    with open(target_dir, "w", encoding="utf-8") as f:
        f.write("\t".join(first_line) + "\n")
        for (xmin, xmax), values in data.items():
            f.write("\t".join([xmin, xmax] + values) + "\n")


if __name__ == '__main__':
    search_path = os.path.join(data_path, "*/*.TextGrid")
    for file in tqdm.tqdm(glob.glob(search_path)):
        direct_dir, filename = get_folder_file_name(file)
        data, name_list = transform_textgrid(file)
        write(data, name_list, os.path.join(save_path, direct_dir + "__" + filename + ".txt"))
