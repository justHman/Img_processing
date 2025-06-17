import numpy as np
import cv2 
import matplotlib.pyplot as plt

def gaussian_kernel(size: int, sigma: float):
    """Tạo kernel Gaussian 2D"""
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    print(x)
    print(y)
    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(- (x**2 + y**2) / (2 * sigma**2))
    print(kernel)
    print(kernel.sum())
    print(kernel/kernel.sum())
    exit(0)
    return kernel   # Chuẩn hóa cho tổng bằng 1

def gamma_sharp(image, gamma):
    hsv = cv2.cv2tColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = 255 * ((hsv[:, :, 2] / 255) ** gamma)
    img = cv2.cv2tColor(hsv, cv2.COLOR_HSV2BGR)
    return img.astype(np.uint8)

def test(arr):
    return arr ** 2

if __name__ == '__main__':
    # kernel = gaussian_kernel(3, 1.0)
    # print(np.round(kernel, 6))  
    # Đọc ảnh
    path = r'D:\Code\Computer_vision\pic\Hoa_giay.jpg'

    # Đọc ảnh xám
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Áp dụng lọc trung bình (mean filter) với kernel 3x3
    mean_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Hiển thị ảnh gốc và ảnh sau khi lọc
    plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Gốc')
    plt.subplot(1, 2, 2), plt.imshow(mean_filtered, cmap='gray'), plt.title('Lọc trung bình')
    plt.show()
