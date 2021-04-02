import cv2
import numpy as np


# 返回阈值和二值图
def threshTwoPeaks(image):
    # 计算回复直方图
    hist = cv2.calcHist([image], [0], None, [256], [1, 255])
    hist[:20, :] = 0
    # 找到灰度直方图的最大峰值对应的灰度值
    # new_hist = hist[140:255]
    new_hist = hist
    maxLoc = np.where(new_hist == np.max(new_hist))
    print(maxLoc[0])
    if len(maxLoc[0]) >= 2:
        firstPeak = maxLoc[0][-1]
        print("第一大峰值，超过1个")
    else:
        firstPeak = maxLoc[0]
        print("第一大峰值，只有一个")
    print("最大灰度值", firstPeak)
    # 寻找灰度直方图的 " 第二个峰值 " 对应的灰度值
    measureDists = np.zeros([115], np.float32)
    for k in range(115):
        # 距离公式，比绝对值扩大的更厉害，更加有利于知道第二个峰值
        measureDists[k] = pow(k - firstPeak, 2) * new_hist[k]
    maxLoc2 = np.where(measureDists == np.max(measureDists))
    if len(maxLoc2[0]) >= 2:
        secondPeak = maxLoc2[0][-1]
        print("第二大峰值，超过一个")
    else:
        secondPeak = maxLoc2[0]
        print("第二大峰值，只有一个")
    print("第二大峰值", secondPeak)
    thresh = 0
    # 找到两个峰值之间的最小值对应的灰度值，作为阈值
    if firstPeak > secondPeak:
        temp = new_hist[int(secondPeak):int(firstPeak)]
        minLoc = np.where(temp == np.min(temp))
        if len(minLoc[0]) >= 2:
            # thresh = secondPeak + minLoc[0][-1] + 140
            thresh = secondPeak + minLoc[0][-1]

        else:
            # thresh = secondPeak + minLoc[0] + 140
            thresh = secondPeak + minLoc[0]

    else:
        if firstPeak == secondPeak:
            # thresh = firstPeak + 140
            thresh = firstPeak
        else:
            temp = new_hist[int(firstPeak):int(secondPeak)]
            minLoc = np.where(temp == np.min(temp))
            if len(minLoc[0]) >= 2:
                # thresh = firstPeak + minLoc[0][-1] + 140
                thresh = firstPeak + minLoc[0][-1]
            else:
                # thresh = firstPeak + minLoc[0] + 140
                thresh = firstPeak + minLoc[0]
    print("阈值：", thresh)
    # 找到阈值后进行阈值处理，得到二值图
    threshImage_out = image.copy()
    threshImage_out[threshImage_out >= thresh] = 255
    threshImage_out[threshImage_out < thresh] = 0
    return thresh, threshImage_out, new_hist
