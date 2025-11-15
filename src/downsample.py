import numpy as np
from correlate_2d import correlate_2d

def downsample(img, factor):
    return img[::factor, ::factor]

def gaussian_kernel(kernel_size, sigma):
    x = np.reshape(np.arange(-kernel_size//2 + 1, kernel_size//2 + 1), (-1,1))
    gaussian_1d = np.exp(-0.5 * (x / sigma)**2)

    # By the separability principle of gaussian kernel
    gaussian_2d = gaussian_1d @ gaussian_1d.T
    return gaussian_2d / np.sum(gaussian_2d)

def filter_downsample(img, factor):
    '''Antialias filtered image before donwsampling'''
    sigma = 2 * factor / 6.0 # from scikit-image documentation
    kernel_size = 2 * factor + 1 # to keep kernel size odd
    kernel = gaussian_kernel(kernel_size, sigma)
    img_filtered = correlate_2d(img, kernel)
    return img_filtered[::factor, ::factor]