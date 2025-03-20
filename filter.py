import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def gamma_sharp(image, gamma):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hsv[:, :, 2] = 255 * ((hsv[:, :, 2] / 255) ** gamma)
    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return img.astype(np.uint8)

def test(arr):
    return arr ** 2

if __name__ == '__main__':
    # Đọc ảnh
    path = r'D:\Code\Computer_vision\pic\Hoa_giay.jpg'
    img = cv.imread(path)
    img_sharped = gamma_sharp(img, 1)
    cv.imshow("Original Image", img)
    cv.imshow("Gamma Sharped Image", img_sharped)
    cv.waitKey(0)
