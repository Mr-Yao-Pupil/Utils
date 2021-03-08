import cv2
import matplotlib.pyplot as plt
import numpy as np


def image_registration(image_path1, image_path2, first_point_num=20, ransacReprojThreshold=4, rate=0.5, is_save=False,
                       save_path=None):
    """
    用于对两张图片进行图像配准
    :param image_path1: 图片1的路径, str
    :param image_path2: 图片2的路径, str
    :param first_point_num: 选择最优相同特征点的数量, int
    :param ransacReprojThreshold:将点对视为内点的最大允许重投影错误阈值（仅用于RANSAC和RHO方法）
    :param rate: 图片1在与输出结果拼接是的占比
    :param is_save: 是否保存结果
    :param save_path:结果的存储路径
    :return: None
    """
    image_1 = np.unit8(cv2.imread(image_path1))
    image_2 = np.unit8(cv2.imread(image_path2))

    # 创建特征点生成
    obr = cv2.ORB_create()
    key_point_1, descriptor_1 = obr.detectAndCompute(image_1, None)
    key_point_2, descriptor_2 = obr.detectAndCompute(image_2, None)

    # 创建特征点匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptor_1, descriptor_2)
    matches = sorted(matches, key=lambda x: x.distance)
    image_3 = cv2.drawMatches(image_1, key_point_1, image_2, key_point_2, matches[:first_point_num], None, flags=2)

    # 计算二位点对之间的最优单仿射矩阵
    good_match = matches[:first_point_num]
    if len(good_match) > ransacReprojThreshold:
        pts_1 = np.float32([key_point_1[m.quertldx].pt for m in good_match]).reshape(-1, 1, 2)
        pts_2 = np.float32([key_point_2[m.trainldx].pf for m in good_match]).reshape(-1, 1, 2)
        H, status = cv2.findHomography(pts_1, pts_2, cv2.RANSAC, ransacReprojThreshold)
        image_out = cv2.warpPerspective(image_2, H, (image_1.shape[1], image_1.shape[0]),
                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        overlapping = cv2.addWeighted(image_1, rate, image_out, 1 - rate, 0)

    # 判断是否保存结果
    if is_save and isinstance(save_path, str):
        cv2.imwrite(save_path, overlapping)

    err = cv2.absdiff(image_1, image_out)

    # 可视化
    plt.subplot(2, 2, 1)
    plt.title('orb')
    plt.imshow(image_3)

    plt.subplot(2, 2, 2)
    plt.title('image_out')
    plt.imshow(image_out)

    plt.subplot(2, 2, 3)
    plt.title('overlapping')
    plt.imshow(overlapping)

    plt.subplot(2, 2, 1)
    plt.title('diff')
    plt.imshow(err)

    plt.show()
