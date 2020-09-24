import cv2
import numpy as np
import time
from skimage.util.shape import view_as_windows

def flattenConvWindows(img0, hfilt):
    img0 = img0.astype('float32')
    hfilt = hfilt.astype('float32')

    # Define size of image and filter
    rows = img0.shape[0]
    cols = img0.shape[1]
    filter_rows = hfilt.shape[0]
    filter_cols = hfilt.shape[1]

    # Pad img0 with values from the image boundary
    pad_height = int((filter_rows-1)/2)
    pad_width = int((filter_cols-1)/2)
    img0 = np.pad(img0, (pad_height,pad_width), 'edge')

    output = []
    for i in range(rows*cols):
        m = int(i/cols)
        n = i%cols
        window = img0[m:m+filter_rows,n:n+filter_rows]
        output.append(window.flatten())
    return np.transpose(np.array(output))

def myImageFilterX(img0, hfilt):
    start_time = time.time()
    img1 = np.dot(hfilt.flatten(), flattenConvWindows(img0, hfilt)).reshape(img0.shape)
    # print('myImageFilterX Runtime:', time.time()-start_time)
    return img1
