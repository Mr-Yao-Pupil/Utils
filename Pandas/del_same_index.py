import pandas


def del_same_index(df, keep='first'):
    """

    :param df: DataFrame, 需要去除相同索引的dataframe
    :param keep: {'first', 'last', False}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Mark duplicates as ``True`` except for the first
              occurrence.
            - 'last' : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.
    :return: df, 删除相同索引的Dataframe
    """

    df = df[~df.index.duplicated(keep=keep)]
    return df
