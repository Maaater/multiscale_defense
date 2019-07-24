import numpy as np
import torch
from torchvision import datasets, transforms
import os
from mnist.utils import template, img_conv2d
from PIL import Image

# 1. make augemented data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=1, shuffle=False)


ns = 5

print("Step1: Using Gaussian Kernels to evaluate training images:")
print("")
for idx, (data, target) in enumerate(train_loader):
    data = data.numpy()[0][0] * 255
    target = target.numpy()[0]

    root = 'data/augemented_data/' + str(target)
    if os.path.exists(root) is False:
        os.makedirs(root)

    # for s in range(ns):
    #     temp = template(2, s * 0.4)
    #     img = img_conv2d(data, temp)
    #     file_name = 'data/augemented_data/' + str(target) + '/' + str(idx) + '-' + str(s) + '.bmp'
    #     im = Image.fromarray(np.uint8(img)).convert('L')
    #     im.save(file_name)

    if idx % 6000 == 0:
        print('Prossing : {}/{} ({:.0f}%)'.format(idx, len(train_loader.dataset),100. * idx / len(train_loader)))

print("Done! The augmentated images are stored in data/augmented_data for next use in step2!")
