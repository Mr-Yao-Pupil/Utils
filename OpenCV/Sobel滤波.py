import cv2


def sobel_conv(image, kernel_size):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imshow('origin', image)
    cv2.waitKey(0)

    sobelXY = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=kernel_size)
    cv2.imshow('grad=1', sobelXY)
    cv2.waitKey(0)
    return sobelXY