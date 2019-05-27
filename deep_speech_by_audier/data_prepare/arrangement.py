""" 操作原始数据集，生成可用的结构化标注文件 """

import os
import glob
import tqdm
import argparse


def _get_partition_files(dirname: str):
    return set([os.path.splitext(file)[0] for file in os.listdir(dirname) if file.endswith(".wav")])


def _transform_thchs30(source_dir: str, output_file: str):
    """
    解析 THCHS30 开源数据集，生成结构化标注信息
    :param source_dir: 下载好的 data_thchs30文件夹地址
    :param output_file: 输出的结构化文本文件
    """
    data_dir, train_dir, test_dir, dev_dir = [os.path.join(source_dir, x) for x in ["data", "train", "test", "dev"]]
    train_sets, test_sets, dev_sets = [_get_partition_files(x) for x in [train_dir, test_dir, dev_dir]]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fw:
        for file in tqdm.tqdm(glob.glob(os.path.join(data_dir, "*.trn"))):
            wav_name = os.path.splitext(os.path.basename(file).strip(".trn"))[0]
            with open(file, "r", encoding="utf-8") as fr:
                content, pinyin, *_ = fr.readlines()
            content = content.strip().replace(" ", "")
            pinyin = pinyin.strip()

            partition = ""
            if wav_name in train_sets:
                partition = "train"
            elif wav_name in test_sets:
                partition = "test"
            elif wav_name in dev_sets:
                partition = "dev"
            fw.write("\t".join([file, content, pinyin, partition]) + "\n")
    print("Transformed Done! Successfully written into %s" % output_file)


def transform(args):
    if args.thchs30:
        source_dir, *output_file = args.thchs30
        if output_file:
            output_file = output_file[0]
        else:
            output_file = os.path.join(source_dir, "thchs30.txt")
        _transform_thchs30(source_dir, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thchs30", type=str, help="where to load thchs source and where to sink result", nargs="*")
    parser.add_argument("--aishell", type=str, help="where to load aishell source and where to sink result", nargs="*")

    args = parser.parse_args()
    transform(args)
    # python arrangement.py --thchs30 "F:/for learn/data source/SpeechCHS/THCHS30/data_thchs30"
