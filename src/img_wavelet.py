import pywt
import numpy as np

def img_wavelet(img, level):
    wv_coef = pywt.wavedec2(img, 'haar', level=level)
    return wv_coef[0]
