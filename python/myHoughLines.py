import numpy as np

def nonMaximumSuppression(H):
    rows = H.shape[0]
    cols = H.shape[1]
    window_size = 151
    pad_size = int((window_size-1)/2)

    H_new = np.zeros(H.shape)
    HPadded = np.pad(H, (pad_size,pad_size), 'constant', constant_values=(0, 0))
    for i in range(rows*cols):
        m = int(i/cols)
        n = i%cols
        window = HPadded[m:m+window_size,n:n+window_size]
        max_index = window.argmax()
        window_row = int(max_index/window_size)
        window_col = max_index%window_size
        row = m+window_row-pad_size
        col = n+window_col-pad_size
        H_new[row,col] = H[row,col]

    return H_new

def myHoughLines(H, nLines):
    # Your implemention
    H = nonMaximumSuppression(H)

    rhos = np.zeros((nLines,))
    thetas = np.zeros((nLines,))
    values = np.zeros((nLines,))
    for theta in range(H.shape[0]):
        for rho in range(H.shape[1]):
            value = H[theta,rho]
            min_index = np.argmin(values)
            if value > values[min_index]:
                values[min_index] = value
                thetas[min_index] = theta
                rhos[min_index] = rho

    return rhos,thetas
