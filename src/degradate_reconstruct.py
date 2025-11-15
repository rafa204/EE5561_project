import numpy as np

def transform_downsample(img, percentage):
    '''Keep just percentage of rows in frequency domain'''
    rows, cols = img.shape
    rows_to_keep = int(np.ceil(percentage*rows))
    img_fourier = np.fft.fftshift(np.fft.fft2(img))
    
    rows_list = np.arange(rows)
    np.random.shuffle(rows_list)
    rows_shuffled = rows_list[:rows_to_keep]

    img_fourier_down = np.ones((rows, cols), dtype=complex) * 1e-15
    mask = np.zeros((rows, cols))

    for row in range(rows_to_keep):
        img_fourier_down[rows_shuffled[row],:] = img_fourier[rows_shuffled[row],:]
        mask[rows_shuffled[row],:] = 1

    return img_fourier_down, mask


