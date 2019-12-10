import numpy as np
import math
from scipy import signal

# 高斯的计算公式
def calc(x, y, sigma):
    res1 = 1 / (2 * math.pi * sigma * sigma)
    res2 = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return res1 * res2

# 得到滤波模版
def template(radius, sigma):
    sideLength = radius * 2 + 1
    result = np.zeros((sideLength, sideLength))
    if sigma == 0:
        result[radius, radius] = 1
        return result
    for i in range(sideLength):
        for j in range(sideLength):
            result[i, j] = calc(i - radius, j - radius, sigma)
    all = result.sum()
    return result / all

def img_conv2d(img, filter):
    img = np.array(img).astype('float')
    if len(img.shape) == 2:
        img = signal.convolve2d(img, filter, mode='same', boundary='symm')
        return img
    if len(img.shape) == 3:
        for i in range(img.shape[2]):
            imc = img[:,:,i]
            imc = signal.convolve2d(imc, filter, mode='same', boundary='symm')
            img[:,:,i] = imc
        return img