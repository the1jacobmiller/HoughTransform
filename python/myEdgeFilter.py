import cv2
import numpy as np
import numpy.matlib as npm
import math

from matplotlib import pyplot as plt
import matplotlib

from GaussianKernel import Gauss2D

from myImageFilter import myImageFilter
from myImageFilterX import myImageFilterX

sobel_x = 1./8.*np.array([[1,    0,   -1],
                        [2,    0,   -2],
                        [1,    0,   -1]],dtype='float32')

sobel_y = 1./8.*np.array([[1,    2,   1],
                        [0,    0,   0],
                        [-1,  -2,  -1]],dtype='float32')

def nonMaximumSuppression(ImEdge, Io):
    rows = ImEdge.shape[0]
    cols = ImEdge.shape[1]

    filter_size = 2
    ImOut = np.zeros(ImEdge.shape)
    ImEdge = np.pad(ImEdge, (filter_size,filter_size), 'constant', constant_values=(0, 0))
    for row in range(filter_size,rows+filter_size):
        for col in range(filter_size,cols+filter_size):
            # round orientation to nearest 0, 45, 90, 135
            orientations = np.array([0., np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            orientation = Io[row-filter_size,col-filter_size]
            if orientation < 0:
                orientation += np.pi
            bucket = np.abs(orientations-orientation).argmin()
            orientation = orientations[bucket]

            # compare with neighbors
            value = ImEdge[row,col]
            if (bucket == 0 or bucket == 4) and (value >= ImEdge[row,col-1] and value and ImEdge[row,col+1] and \
            value >= ImEdge[row,col-2] and value >= ImEdge[row,col+2]):
                ImOut[row-filter_size,col-filter_size] = value
            elif bucket == 1 and (value >= ImEdge[row+1,col-1] and value >= ImEdge[row-1,col+1] and \
            value >= ImEdge[row+2,col-2] or value >= ImEdge[row-2,col+2]):
                ImOut[row-filter_size,col-filter_size] = value
            elif bucket == 2 and (value >= ImEdge[row-1,col] and value >= ImEdge[row+1,col] and \
            value >= ImEdge[row-2,col] and value >= ImEdge[row+2,col]):
                ImOut[row-filter_size,col-filter_size] = value
            elif bucket == 3 and (value >= ImEdge[row-1,col-1] and value >= ImEdge[row+1,col+1] and \
            value >= ImEdge[row-2,col-2] and value >= ImEdge[row+2,col+2]):
                ImOut[row-filter_size,col-filter_size] = value
    return ImOut

def myEdgeFilter(img0, sigma):
	# Your implemention

    # Smooth image with Gaussian filter
    hsize = 2*math.ceil(3*sigma)+1
    hfilt = Gauss2D((hsize,hsize),sigma)
    img1 = myImageFilterX(img0, hfilt)

    # Calculate image gradients
    Ix = myImageFilterX(img1, sobel_x)
    Iy = myImageFilterX(img1, sobel_y)

    # Calculate edge magnitude image
    ImEdge = np.sqrt(np.square(Ix) + np.square(Iy))

    # Calculate edge orientation image
    Io = np.arctan2(Iy, Ix)

    # Non-maximum suppression
    ImEdge = nonMaximumSuppression(ImEdge, Io)

    return ImEdge,Io,Ix,Iy
