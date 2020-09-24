import cv2
import numpy as np
import time


def myImageFilter(img0, hfilt):
    # Your implemention
    start_time = time.time()

    img0 = img0.astype('float32')
    hfilt = hfilt.astype('float32')
    # Define size of image and filter
    rows = img0.shape[0]
    cols = img0.shape[1]
    filter_rows = hfilt.shape[0]
    filter_cols = hfilt.shape[1]
    img1 = np.zeros(img0.shape,dtype='float32')

    # Pad img0 with values from the image boundary
    pad_height = int((filter_rows-1)/2)
    pad_width = int((filter_cols-1)/2)
    img0 = np.pad(img0, (pad_height,pad_width), 'edge')

    for m in range(rows):
        for n in range(cols):
            img1[m,n] = np.sum(hfilt*img0[m:m+filter_rows,n:n+filter_cols])

    # print('myImageFilter Runtime:', time.time()-start_time)

    return img1
