# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:39:44 2019

@author: Cigar
"""

import os
import glob
import tqdm
from zipfile import ZipFile
from unrar.rarfile import RarFile
from project_trial.constant import PATH


def extract_file(file, target_dir):
    if file.endswith("rar"):
        pack = RarFile(file)
    elif file.endswith("zip"):
        pack = ZipFile(file)
    else:
        raise Exception("The valid file should be *.rar or *.zip")
    print("start extracting file %s" % os.path.basename(file))
    pack.extractall(path=target_dir)
    print("%s extracted successfully!" % os.path.basename(file))


if __name__ == "__main__":
    for file in tqdm.tqdm(glob.glob(PATH + "*.*[zip, rar]")):
        extract_file(file, PATH)

