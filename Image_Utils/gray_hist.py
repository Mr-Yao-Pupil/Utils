import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def calc_gray_hist(image):
    """
    获取图片的灰度值向量
    :param image: 图片的灰度图矩阵
    :return: 长度为255的灰度分布向量
    """
    w, h = image.shape
    gray_hist = np.zeros([256], np.uint64)
    for x in range(w):
        for y in range(h):
            if image[x][y] != 0:
                gray_hist[image[x][y]] += 1
    return gray_hist


def get_gray_hist(image_path, save=False, save_path=None):
    """
    原理
    :param image_path: 需要计算灰度直方图的图片路径
    :param save: bool,option.是否保存, 默认不保存，如若要保存请补全save_path
    :param save_path: str.如果save为True必填.
    :return:
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2GRAY)
    gray_hist = calc_gray_hist(image)
    plt.plot(range(256), gray_hist, 'r', linewidth=2, c='black')
    plt.axis([0, 255, 0, np.max(gray_hist)])
    plt.xlabel('gray_leave')
    plt.ylabel('number_of_pixels')

    if save:
        if not save_path:
            raise ValueError(f"路径错误,检查路径:{save_path}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, os.path.basename(image_path)))
    plt.show()


def gray_hist(image_path, save=False, save_path=None):
    """
    灰度计算
    :param image_path: 需要计算灰度直方图的图片路径
    :param save: bool,option.是否保存, 默认不保存，如若要保存请补全save_path
    :param save_path: str.如果save为True必填.
    :return:
    """
    img = cv2.imread(image_path, 0)
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("number of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    if save:
        if not save_path:
            raise ValueError(f"路径错误,检查路径:{save_path}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, os.path.basename(image_path)))
    plt.show()
