
"""统计各个textgrid文件的结构，梳理是否结构一致"""

import os
import glob
import tqdm
import numpy as np
import pandas as pd
from project_trial.constant import SAVE_PATH

EXPECTED_COLUMNS = ['xmin', 'xmax', '文本', '性別', '身份', '噪声']


if __name__ == '__main__':
    search_path = os.path.join(SAVE_PATH, "*.txt")
    not_satisfied = []
    for file in tqdm.tqdm(glob.glob(search_path)):
        try:
            with open(file, "r", encoding="utf-8") as f:
                df = pd.read_table(f)
            if not np.alltrue(EXPECTED_COLUMNS == df.columns):
                print("%s columns not satisfied!" % file)
                not_satisfied.append(file)
        except Exception as e:
            raise Exception("%s error, details: %s" % (file, e))

    if not_satisfied:
        print("%d files not satisfied!" % len(not_satisfied))
        with open(os.path.join(SAVE_PATH, "!invalid_files.dat"), "r") as f:
            f.write("\n".join(not_satisfied))
    else:
        print("All files satisfied!")
