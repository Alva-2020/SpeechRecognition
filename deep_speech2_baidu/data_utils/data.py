
import pandas as pd


def read_labeled_data(data_file: str) -> pd.DataFrame:
    """Read saved data from `prepare.py`
    :param data_file: String, the src path to file
    :return: a pandas.dataframe
    """
    columns = ["src", "duration", "content", "pinyin", "data_type", "data_source"]
    return pd.read_csv(data_file, sep="\t", engine="python", encoding="utf-8", header=None, names=columns)
