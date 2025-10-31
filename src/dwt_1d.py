import numpy as np
import math

def corr_1d(signal, kernel):
    """Implementing 'same' correlation"""
    m = signal.shape[0]
    n = kernel.shape[0]

    padded = np.zeros(m+n-1)
    padded[0:m] = signal
    padded = np.roll(padded, math.ceil((n-1)/2))

    index = np.arange(m)[:,None] + np.arange(n)[None,:]
    block = padded[index.flatten()].reshape((m,n))
    result = np.sum(block * kernel, axis=-1).flatten()
    return result

def dwc_1d(sig, levels, scaling_function, wavelet_function):
    """
    Returns a list of np.arrays. [approx., detail]
    If legnth of signal do not allow to reach all levels, truncate 
    """
    # Check if odd
    if len(sig) % 2 != 0:
        sig = np.append(sig, 0)
    
    X = []
    for i in range(levels):
        # "double inversion" means simple correlation
        approx = corr_1d(sig, scaling_function)
        detail = corr_1d(sig, wavelet_function)
        # Decimate
        #breakpoint()
        approx = approx[::2]
        detail = detail[::2]
        X.append({"approximation": approx, "detail": detail})

        # Peparation of the next loop
        sig = approx.copy()
        if len(sig) % 2 != 0:
            sig = np.append(sig, 0)

        if len(approx) <= len(scaling_function):
            break
    return X


if __name__ == "__main__":
    import pywt

    sig = np.array([3, 1, 0, 4, 8, 6, 9, 8, 4 , 4, 5, 6 ,3])
    cA, cD = pywt.dwt(sig, 'haar', mode='zero')

    print("Approximation:", cA)
    print("Detail:", cD) 

    scaling_function = (1/np.sqrt(2)) * np.array([1, 1])
    wavelet_function =  (1/np.sqrt(2)) * np.array([1, -1])
    levels = 1
    X = dwc_1d(sig, levels, scaling_function, wavelet_function)
    print(X)






