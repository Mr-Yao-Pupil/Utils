import cv2
import matplotlib.pyplot as plt
import numpy as np


def two_mode_method(image, thresh):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_list = image.ravel()
    image_list = image_list[image_list > 20]
    # plt.hist(image.ravel(), 256)
    plt.hist(image_list, 256)

    ret1, th1 = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    cv2.imshow("image", image)
    cv2.imshow('th1', th1)
    masked = cv2.add(image, np.zeros(np.shape(image), dtype=np.uint8), mask=th1)
    cv2.imshow('mask_ori', masked)
    cv2.waitKey(0)
