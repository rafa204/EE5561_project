import numpy as np

def downsample(img, factor):
    return img[::factor, ::factor]