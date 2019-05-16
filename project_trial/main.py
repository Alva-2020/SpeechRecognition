# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:27:17 2019

@author: Cigar
"""
import sys
sys.path.append("F:/Code projects/Python/SpeechRecognition/")

from project_trial.utils import textgrid

file = r"F:\for learn\asr_trial_data\BJ-9.53H\20180705_1530792985_3799813.TextGrid"

text = textgrid.TextGrid.load(file)
