import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import filter

# Đọc ảnh
path = r'D:\Code\Computer_vision\pic\Hoa_giay.jpg'
img = cv.imread(path)

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_sharped = filter.gamma_sharp(img_rgb, 0.1)
cv.imshow("Original Image", img_rgb)
cv.imshow("Gamma Sharped Image", img_sharped)
cv.waitKey(0)