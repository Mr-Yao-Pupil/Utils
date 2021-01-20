import pandas as pd


def function_same(a, b):
    if a == b:
        return True
    else:
        return False


def del_diffrient_series(df, columns_name1, columns_name2):
    """
    删除dataframe同行两列取值不相同的数据
    :param df: 待处理的DataFrame实例对象
    :param columns_name1: 对比的第一列索引
    :param columns_name2: 对比的第二列索引
    :return: 删除两列索引的不同行后的dataframe
    """
    if columns_name1 not in df.columns.tolist():
        raise ValueError(f"{columns_name1}不是输入Dataframe中的列索引")
    if columns_name2 not in df.columns.tolist():
        raise ValueError(f"{columns_name2}不是输入Dataframe中的列索引")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"参数df必须输入pandas.DataFrame的实例对象")
    df = df[df.apply(lambda x: function_same(x[columns_name1], x[columns_name2]), axis=1)]
    return df


def del_same_series(df, columns_name1, columns_name2):
    """
    删除dataframe同行两列取值相同的数据
    :param df: 待处理的DataFrame实例对象
    :param columns_name1: 对比的第一列索引
    :param columns_name2: 对比的第二列索引
    :return: 删除两列索引的相同行后的dataframe
    """
    if columns_name1 not in df.columns.tolist():
        raise ValueError(f"{columns_name1}不是输入Dataframe种的列索引")
    if columns_name2 not in df.columns.tolist():
        raise ValueError(f"{columns_name2}不是输入Dataframe种的列索引")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"参数df必须输入pandas.DataFrame的实例对象")
    df = df[df.apply(lambda x: not function_same(x[columns_name1], x[columns_name2]), axis=1)]
    return df
