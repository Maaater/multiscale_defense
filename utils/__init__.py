import numpy as np 
from scipy import signal

def img_conv2d(img, filter):
    img = np.array(img).astype('float')
    # img = np.array(img)
    if len(np.shape(img)) == 2:
        return signal.convolve2d(img, filter, mode='same', boundary='symm')
    if np.shape(img)[2] == 3:
        img[:,:,0] = signal.convolve2d(img[:,:,0], filter, mode='same', boundary='symm')
        img[:,:,1] = signal.convolve2d(img[:,:,1], filter, mode='same', boundary='symm')
        img[:,:,2] = signal.convolve2d(img[:,:,2], filter, mode='same', boundary='symm')
        return img

def denoiser(denoiser_name, img,sigma):
    from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means, wiener)
    if denoiser_name == 'wavelet':
        return denoise_wavelet(img,sigma=sigma, mode='soft', multichannel=False,convert2ycbcr=True, method='BayesShrink')
    elif denoiser_name == 'TVM':
        return denoise_tv_chambolle(img, multichannel=True)
    elif denoiser_name == 'bilateral':
        return denoise_bilateral(img, bins=1000, multichannel=True)
    elif denoiser_name == 'deconv':
        return wiener(img)
    elif denoiser_name == 'NLM':
        return denoise_nl_means(img, multichannel=True)
    else:
        raise Exception('Incorrect denoiser mentioned. Options: wavelet, TVM, bilateral, deconv, NLM')

import numpy as np 
from PIL import Image
import scipy.signal as signal

def medfilt2x2(image):
    new_img = np.zeros(image.shape)
    img = np.pad(image, ((1, 0), (1, 0)), 'reflect')
    
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            patch = np.reshape(img[i:i+2, j:j+2], (4,))
            new_img[i,j] = np.sort(patch)[2]
    return new_img


def median2x2(image):
    img = np.array(image)
    img = medfilt2x2(img)
    return img


def median3x3(image):
    img = np.array(image)
    img = signal.medfilt2d(img, (3, 3))
    return img
