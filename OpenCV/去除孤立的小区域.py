import cv2

def IsoArea(image, fill_color=(0, 0, 0)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    retval, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    threshold = 20 * 20
    contours, hierarch = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # 计算轮廓所占面积
        if area < threshold:  # 将area小于阈值区域填充背景色，由于OpenCV读出的是BGR值
            cv2.drawContours(image, [contours[i]], -1, fill_color, thickness=-1)  # 原始图片背景BGR值(84,1,68)
            continue

    cv2.imshow('test', image)
    cv2.waitKey()
    return image