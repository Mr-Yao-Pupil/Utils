# 通过预测模型检测异常数据
import pandas as pd
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt


def find_outliers(model, X, y, sigma=3, savepath="outliers.png"):
    """
    通过模型检测异常点，可以使用岭回归之类的分类模型
    :param model: 模型实例
    :param X: DataFrame，训练数据
    :param y: Series, 标签
    :param sigma: 范围
    :param savepath: plt可视化的存储路径
    :return: 包含异常点信息的DataFrame
    """
    # 建立模型预测的y值和模型预测的y值的Series
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    except:
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    # 计算y值和预测值之间的均值和标准差
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # 对数据进行标准化操作后找出异常值
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    # 绘制结果的plot
    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, ".")
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred')

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred')

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'outlier'])
    plt.xlabel('z')
    plt.savefig(savepath)

    # 返回其他异常点信息
    return pd.Series({"R2": model.score(X, y),
                      "mse_distance": mean_squared_error(y, y_pred),
                      "mean_of_residuals": mean_resid,
                      "std_of_residuals": std_resid,
                      "number_of_outliers": len(outliers),
                      "outliers": outliers.tolist()})
