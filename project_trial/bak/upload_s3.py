# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:24:18 2019

@author: Cigar
"""
import os
import glob
import tqdm
import threading
from evan_utils.conn import get_aws_conn
from project_trial.constant import FRAGMENTED_AUDIO_PATH


UPLOADED_STATUS_LOG = os.path.join(FRAGMENTED_AUDIO_PATH, "uploaded.log")
BUCKET = "zhaochengming"
session = get_aws_conn(region_name="us-east-2")
s3 = session.resource("s3")


def write_uploaded_index(uploaded_index: int):
    with open(UPLOADED_STATUS_LOG, "w") as f:
        f.write("%s\n" % uploaded_index)
    print("Uploaded Index: %d" % uploaded_index)


def get_uploaded_index():
    if os.path.exists(UPLOADED_STATUS_LOG):
        with open(UPLOADED_STATUS_LOG, "r") as f:
            return int(f.readlines()[0].strip())

    write_uploaded_index(-1)
    return -1


def upload(file):
    session = get_aws_conn(region_name="us-east-2")
    s3 = session.resource("s3")
    _, dir_file = file.split(FRAGMENTED_AUDIO_PATH)
    object_file = dir_file.replace("\\", "/").strip("/")
    s3.meta.client.upload_file(file, BUCKET, object_file)
    return file


if __name__ == "__main__":

    # # single thread
    # session = get_aws_conn(region_name="us-east-2")
    # s3 = session.resource("s3")
    # uploaded_index = get_uploaded_index()
    # i = -1
    # for file in tqdm.tqdm(glob.glob(os.path.join(FRAGMENTED_AUDIO_PATH, "*/*.wav"))):
    #     i += 1
    #     if i < uploaded_index:
    #         continue
    #     _, dir_file = file.split(FRAGMENTED_AUDIO_PATH)
    #     object_file = dir_file.replace("\\", "/").strip("/")
    #     try:
    #         response = s3.meta.client.upload_file(file, BUCKET, object_file)
    #         if not response:
    #             uploaded_index = i
    #     except Exception as e:
    #         write_uploaded_index(uploaded_index)

    # # multi-thread
    for file in tqdm.tqdm(glob.glob(os.path.join(FRAGMENTED_AUDIO_PATH, "*/*.wav"))):
        t = threading.Thread(target=upload, args=(file, ))
        t.start()



