import json
import os
import re
from typing import LiteralString

import pandas as pd
import numpy as np
import csv
from dpm.utils import log
import yaml
import joblib
import openpyxl

logger = log.logger()


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_xlsx(path, sheet_name):
    with open(path, "rb") as x:
        sheet = pd.read_excel(x, sheet_name=sheet_name, header=0)
    return sheet


def save_xlsx(df, path, sheet_name_, index_=False, mode_="wa", if_sheet_exists_="replace"):
    """

    :param df:
    :param path:
    :param sheet_name_:
    :param index_:
    :param mode_: str, 'wa' is usr-defined for writing and adding,'w',writing, 'a', adding
    :param if_sheet_exists_:
    :return: None
    """
    match mode_:
        case "wa":
            if file_is_exist(path):
                mode_ = "a"
            else:
                mode_ = "w"
                if_sheet_exists_ = None
        case _:
            mode_ = mode_
    with pd.ExcelWriter(path, engine='openpyxl', mode=mode_, if_sheet_exists=if_sheet_exists_) as writer:
        df.to_excel(writer, sheet_name=sheet_name_, index=index_)
        logger.debug(f"save sheet{sheet_name_} {path} 🍺")


def read_csv(path, header_=0, datatype: dict = None,delimiter_=None) -> pd.DataFrame:
    with open(path, "r") as p:
        if not delimiter_:
            delimiter_ = detect_delimiter(path)
        data = pd.read_csv(p, header=header_, sep=delimiter_, dtype=datatype)
    return data


def save_csv(df: pd.DataFrame, path, _index=False):
    df.to_csv(path, index=_index, sep=",")
    logger.debug(f"save dataframe {path} 🍺")


def read_json(path):
    with open(path, "r") as j:
        data = json.load(j)
    return data


def save_json(file, path):
    with open(path, "w") as j:
        json.dump(file, j)
    logger.debug(f"save json file in {path} 🍺")


def read_joblib(path):
    data = joblib.load(path)
    return data


def save_joblib(file, path):
    joblib.dump(file, path)
    logger.debug(f"save joblib file {path} 🍺")


def save_yaml(file, path):
    with open(path, 'w') as f:
        yaml.dump(file, f)
    logger.debug(f"save yaml file in {path} 🍺")


def read_yaml(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    return data


def save_array(path: str | os.PathLike[str], array: np.array):
    np.save(path, array)
    logger.debug(f"save array file in {path} 🍺")


def read_array(path):
    return np.load(path)


def save_arrays(path, **kwargs):
    """
    保存多个数组到一个 .npz 文件中，键名为变量名。

    参数:
    - path: 保存的文件路径。
    - kwargs: 要保存的数组，自动读取变量名作为键名。

    Example
    --------
    >>> save_arrays("test.npz", array1=array1, array2=array2)
    """
    np.savez_compressed(path, **kwargs)
    logger.debug(f"save arrays file in {path} 🍺")


def read_arrays(path):
    """
    从 .npz 文件中读取数组，返回一个包含数组的字典。

    参数:
    - path: 要读取的文件路径。

    返回:
    - 一个包含数组的字典，可以通过 .files 属性查看键名。

    Example
    --------
    >>> arrays = read_arrays("test.npz")

    """
    return np.load(path, allow_pickle=True)


def detect_delimiter(file_path):
    """检测文件分隔符"""
    with open(file_path, 'r') as file:
        sample = file.readline()  # 读取文件的第一行来测试
        sniffer = csv.Sniffer()
        sniffer.preferred = [';', ',', '\t', ' ']
        dialect = sniffer.sniff(sample)
        return dialect.delimiter


def file_is_exist(path) -> bool:
    """check if defined path has valid file already"""
    return os.path.isfile(path)


def insert_str(old_str: str | LiteralString | bytes, insert_str: str, find_syntax: str) -> str:
    index = old_str.rfind(find_syntax)
    if index == -1:
        raise ValueError(f"{insert_str} not found in {old_str}")
    return old_str[:index] + insert_str + old_str[index:]
