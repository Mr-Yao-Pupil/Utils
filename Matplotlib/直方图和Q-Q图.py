import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd


def draw_one_Histogram_QQchart(data, issave=True, save_path="Histogram_QQchart.png"):
    """
    绘制一条数据的直方图和QQ图
    :param data: Series, 某一个维度的数据信息
    :param issave: 是否存储绘制结果
    :param save_path: 图片存储地址
    :return: None
    """
    if not isinstance(data, pd.Series):
        raise ValueError(f'输入数据类型错误,输入数据不支持{type(data)}, 输入数据类型应为Series')
    ax = plt.subplot(1, 2, 1)
    sns.distplot(data, fit=stats.norm)
    ax = plt.subplot(1, 2, 2)
    stats.probplot(data, plot=plt)
    plt.show()
    if issave:
        plt.savefig(save_path)


def draw_all_Histogram_QQchart(data, data_cols, issave=True, save_path="Histogram_QQchart.png"):
    """
    绘制多条数据的直方图和QQ图
    :param data: DataFrame, 需要绘制数据的巨量
    :param data_cols: 绘制数据的总数,如设为6则至绘制前六列数据的直方图和QQ图
    :param issave: 是否保存绘制的图
    :param save_path: 绘制结果的存储路径
    :return: None
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f'输入数据类型错误,输入数据不支持{type(data)},输入数据类型应为DataFrame')
    data_rows = len(data.columns)
    plt.figure(figsize=(4 * data_cols, 4 * data_rows))

    i = 0
    for col in data.columns:
        # 绘制直方图
        i += 1
        ax = plt.subplot(data_rows, data_cols, i)
        sns.distplot(data[col], fit=stats.norm)

        # 绘制QQ图
        i += 1
        ax = plt.subplot(data_rows, data_cols, i)
        res = stats.probplot(data[col], plot=plt)
    plt.tight_layout()
    plt.show()
    if issave:
        plt.savefig(save_path)