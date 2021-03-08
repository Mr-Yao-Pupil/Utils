import pandas as pd
import os


def is_list(x):
    try:
        x = eval(x)
        if isinstance(x, list):
            return True
        else:
            return False
    except:
        return False


def deal_str_list_csv(csv_path, columns_name):
    df = pd.read_csv(csv_path)
    str_df = df[df[columns_name].map(lambda x: not is_list(x))]
    list_df = df[df[columns_name].map(lambda x: is_list(x))]
    return str_df, list_df
