import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def draw_one_KDEchart(train_data, test_data, data_name, issave=False, save_path="Histogram_QQchart.png"):
    """
    绘制一条数据的KDE分布图
    :param train_data:Series, 训练数据
    :param test_data:Series, 测试数据
    :param data_name: 数据名称，再绘制的图片上显示
    :param issave:是否存储绘制出来的图片
    :param save_path:绘制出来图片存储的地址
    :return:None
    """
    if not isinstance(train_data, pd.Series):
        raise ValueError(f'输入数据类型错误,输入数据不支持{type(train_data)},输入数据类型应为Series')
    if not isinstance(test_data, pd.Series):
        raise ValueError(f'输入数据类型错误,输入数据不支持{type(test_data)},输入数据类型应为Series')

    plt.figure(figsize=(8, 4), dpi=50)
    ax = sns.kdeplot(train_data, color="Red", shade=True)
    ax = sns.kdeplot(test_data, color="Blue", shade=True)
    ax.set_xlabel(str(data_name))
    ax.set_ylabel('Frequency')
    ax = ax.legend(['train', 'test'])
    plt.show()
    if issave:
        plt.savefig(save_path)


def draw_all_KDEchart(train_data, test_data, dist_cols=6, issave=False, save_path="Histogram_QQchart.png"):
    """

    :param train_data: DataFrame, 训练数据
    :param test_data: DataFrame, 测试数据
    :param dist_cols: 绘制数据的总数,如设为6则至绘制前六列数据的直方图和QQ图
    :param issave: 是否保存绘制出来的图片
    :param save_path: 绘制出来图片的存储地址
    :return:
    """
    if not isinstance(train_data, pd.DataFrame):
        raise ValueError(f'输入数据类型错误,输入数据不支持{type(train_data)},输入数据类型应为DataFrame')
    if not isinstance(test_data, pd.DataFrame):
        raise ValueError(f'输入数据类型错误,输入数据不支持{type(train_data)},输入数据类型应为DataFrame')
    dist_rows = len(test_data.columns)
    plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))
    i = 1
    for col in test_data.columns:
        ax = plt.subplot(dist_rows, dist_cols, i)
        ax = sns.kdeplot(train_data[col], color="Red", shade=True)
        ax = sns.kdeplot(test_data[col], color="Blue", shade=True)
        ax.set_xlabel(col)
        ax.set_ylabel("Freqiemcy")
        ax = ax.legend(["train", 'test'])
        i += 1
    plt.show()
    if issave:
        plt.savefig(str(save_path))
