import math
import numpy as np


class GaussianBlur():
    # 初始化
    def __init__(self, radius=1, sigema=1.5):
        self.radius = radius
        self.sigema = sigema

    # 高斯的计算公式
    def calc(self, x, y):
        res1 = 1 / (2 * math.pi * self.sigema * self.sigema)
        res2 = math.exp(-(x * x + y * y) / (2 * self.sigema * self.sigema))
        return res1 * res2

    # 得到滤波模版
    def template(self):
        sideLength = self.radius * 2 + 1
        result = np.zeros((sideLength, sideLength))
        if self.sigema == 0:
            result[self.radius, self.radius] = 1
            return result
        for i in range(sideLength):
            for j in range(sideLength):
                result[i, j] = self.calc(i - self.radius, j - self.radius)
        all = result.sum()
        return result / all

    # 滤波函数
    def filter(self, image, template, mode=None):
        arr = np.array(image)
        height = arr.shape[0]
        width = arr.shape[1]
        depth = arr.shape[2]
        padwidth = 0
        if mode == 'same':  # padding with kernel (size-1)/2
            padwidth = int((template.shape[0] - 1) / 2)
            t = arr
            arr = np.zeros((height + 2 * padwidth, width + 2 * padwidth, depth))
            arr[:, :, 0] = np.pad(t[:, :, 0], ((padwidth, padwidth), (padwidth, padwidth)), 'symmetric')
            arr[:, :, 1] = np.pad(t[:, :, 1], ((padwidth, padwidth), (padwidth, padwidth)), 'symmetric')
            arr[:, :, 2] = np.pad(t[:, :, 2], ((padwidth, padwidth), (padwidth, padwidth)), 'symmetric')

        height = arr.shape[0]
        width = arr.shape[1]
        depth = arr.shape[2]
        newData = np.zeros((height, width, depth))
        for d in range(depth):
            for i in range(self.radius, height - self.radius):
                for j in range(self.radius, width - self.radius):
                    t = arr[i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1, d]
                    a = np.multiply(t, template)
                    newData[i, j, d] = a.sum()
        newImage = newData[padwidth:height - padwidth, padwidth:width - padwidth, :]
        return newImage


# r = 2  # 模版半径，自己自由调整
# s = 5  # sigema数值，自己自由调整
# GBlur = GaussianBlur(radius=r, sigema=s)  # 声明高斯模糊类
# temp = GBlur.template()  # 得到滤波模版
# print(temp)
# from PIL import Image
# im = Image.open('white_wolf.png')  # 打开图片
# image = GBlur.filter(im, temp, 'same')  # 高斯模糊滤波，得到新的图片
# im_out = Image.fromarray(image.astype(np.uint8))
# im_out.show() # 图片显示
