import numpy as np
import math

def corr_1d(signal, kernel):
    """'same' correlation, periodic padding"""
    n = kernel.shape[0]
    spaces = math.ceil((n-1)/2)

    if n % 2 != 0:
        padded = np.pad(signal, (spaces, spaces), mode='wrap')
    else:
        padded = np.pad(signal, (spaces, spaces-1), mode='wrap')
    result = np.correlate(padded, kernel, mode='valid')
    return result

def dwc_1d(sig, levels, scaling_function, wavelet_function):
    """
    Returns a list of np.arrays. [approx., detail]
    If legnth of signal do not allow to reach all levels, truncate 
    """
    # Check if odd
    if len(sig) % 2 != 0:
        sig = np.append(sig, sig[0])
    
    X = []
    for i in range(levels):
        # "double inversion" means simple correlation
        approx = corr_1d(sig, scaling_function)
        detail = corr_1d(sig, wavelet_function)
        # Decimate (start at 1 to match pywt)
        approx = approx[1::2]
        detail = detail[1::2]
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
    sig = np.array([3, 1, 0, 4, 8, 6, 9, 8, 4 , 4, 5, 6 ])
    wavelet = pywt.Wavelet('haar')
    cA = sig.copy()
    levels = 3
    for level in range(levels):
        cA, cD = pywt.dwt(cA, wavelet, mode='periodization')
        print(f"Level {level+1}")
        print("Approximation:", cA)
        print("Detail:", cD)

    scaling_function = (1/np.sqrt(2)) * np.array([1, 1])
    wavelet_function =  (1/np.sqrt(2)) * np.array([1, -1])
    levels = 3
    X = dwc_1d(sig, levels, scaling_function, wavelet_function)
    print('mine:', X)






