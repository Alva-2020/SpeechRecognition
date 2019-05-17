
"""
注：前期已通过 bak/_explore探查过，各个txt file拥有同样的结构
将所有txt合并成一个文件，并增加文件注释
"""

import os
import glob
import tqdm
from project_trial.constant import SAVE_PATH, MERGED_FILE
from typing import Generator

# Already checked satisfied for all files in bak/_explore
_COMMON_COLUMNS = ['xmin', 'xmax', 'content', 'sex', 'role', 'noise']
COLUMNS = ["folder", "file"] + _COMMON_COLUMNS  # add additional columns to indicate data's source file
search_path = os.path.join(SAVE_PATH, "*.txt")


def split_filename(full_path_file: str) -> (str, str):
    basename = os.path.basename(full_path_file)  # txt文件名
    name, _ = os.path.splitext(basename)  # 去掉后缀
    folder, filename = name.split("__")  # 获取对应的文件夹和文件名（不含后缀名）
    return folder, filename


def generate_content(search_path: str) -> Generator:
    for file in tqdm.tqdm(glob.glob(search_path)):
        folder, filename = split_filename(file)
        with open(file, "r", encoding="utf-8") as f:
            _ = next(f)  # ignore first line
            for line in f:
                line = line.strip()
                if line:
                    yield folder + "\t" + filename + "\t" + line


if __name__ == "__main__":
    # Time Cost: < 30s
    with open(MERGED_FILE, "w", encoding="utf-8") as f:
        f.write("\t".join(COLUMNS) + "\n")  # 写入首行
        for line in generate_content(search_path):
            f.write(line + "\n")
