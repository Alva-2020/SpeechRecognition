
import os
import pandas as pd
from typing import Dict, Union, List, Optional


_LABELED_DATA_COLS = ["src", "duration", "content", "pinyin", "data_type", "data_source"]
_NOISE_DATA_COLS = ["src", "duration", "data_source"]
_IMPULSE_DATA_COLS = ["src", "duration", "data_source"]

_TAG_ALIAS = {
    "labeled_data": _LABELED_DATA_COLS,
    "noise": _NOISE_DATA_COLS,
    "impulse": _IMPULSE_DATA_COLS
}

_TAG_NAMES = list(_TAG_ALIAS.keys())


def read_data(data_file: str, data_tag: Optional[str]=None, to_dict: bool=False) -> Union[pd.DataFrame, List[Dict]]:
    """
    Read saved data from `prepare.py`
    :param data_file: String, the src path to file
    :param data_tag: String, the type of data to load. When not specified, this will expect data to have columns.
    :param to_dict: bool, whether output as a List[dict]
    :return: a pandas.dataframe | List[dict]
    """
    if not os.path.isfile(data_file):
        raise IOError("Invalid data source path %s" % data_file)
    if data_tag is None:
        df = pd.read_csv(data_file, sep="\t", engine="python", encoding="utf-8")
    else:
        data_tag = data_tag.lower()
        if data_tag not in _TAG_ALIAS:
            raise ValueError("Unknown tag (%s) to load data. Possible choices are: %s" % (data_tag, " ".join(_TAG_NAMES)))
        columns = _TAG_ALIAS[data_tag]
        df = pd.read_csv(data_file, sep="\t", engine="python", encoding="utf-8", header=None, names=columns)
    if to_dict:
        return list(df.to_dict(orient="index").values())
    return df
