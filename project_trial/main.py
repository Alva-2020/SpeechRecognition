# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:27:17 2019

@author: Cigar
"""

import zipfile

path = "F:/for learn/asr_trial_data/"

x = zipfile.ZipFile(path + "BJ33.12+SH10.zip")

infos = x.infolist()
