import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def draw_one_Liner_Regression_chart(data, data_name, issave=False, save_path="Liner_Regression_chart.png"):
    """
    绘制DataFram中单条数据的线性回归图
    :param data: 包含所有数据信息的DataFrame
    :param data_name: DataFrame中某一列的名字
    :param issave: 是否保存绘制的图片
    :param save_path: 绘制保存图片的绘制地址
    :return:
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f'输入数据类型错误,参数X输入数据不支持{type(data)},输入数据类型应为DataFrame')
    if data_name not in data.columns.tolist():
        raise ValueError(f"data中不包含{data_name}信息！")
    if isinstance(issave, bool):
        raise TypeError(f"输入数据类型错误,参数X输入数据不支持{type(issave)},输入数据类型应为Bool")

    plt.figure(figsize=(8, 4), dpi=150)

    ax = plt.subplot(1, 2, 1)
    sns.regplot(x=data_name, y='target', data=data, ax=ax, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                line_kws={'color': 'k'})
    plt.xlabel(data)
    plt.ylabel('target')

    ax = plt.subplot(1, 2, 2)
    sns.distplot(data[data_name].dropna())
    plt.xlabel(data_name)

    if issave and isinstance(save_path, str):
        plt.savefig(save_path)

    plt.show()


def draw_all_Liner_Regression_chart(data, issave=False, save_path="Liner_Regression_charts.png"):
    """
    绘制DataFram中所有数据的线性回归图
    :param data: DataFram, 包含所有数据信息的Pandas
    :param issave: 是否保存绘制的图片
    :param save_path: 绘制图的存储路径
    :return:
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f'输入数据类型错误,参数X输入数据不支持{type(data)},输入数据类型应为DataFrame')
    if isinstance(issave, bool):
        raise TypeError(f"输入数据类型错误,参数X输入数据不支持{type(issave)},输入数据类型应为Bool")
    fcols = 6
    frows = len(data.columns)
    plt.figure(figsize=(5 * fcols, 4 * frows))

    i = 0
    for col in data.columns:
        i += 1
        ax = plt.subplot(frows, fcols, i)
        sns.regplot(x=col, y='target', data=data, ax=ax, scatter_kws={'maker': '.', 's': 3, 'slpha': 0.3},
                    line_kws={'color': 'k'})
        plt.xlabel(col)
        plt.ylabel('target')

        i += 1
        ax = plt.subplot(frows, fcols, i)
        sns.distplot(data[col].dropna())
        plt.xlabel(col)

    if issave:
        plt.savefig(save_path)
    plt.show()
