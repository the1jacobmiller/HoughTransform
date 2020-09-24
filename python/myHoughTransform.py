import cv2
import numpy as np


def myHoughTransform(InputImage, rho_resolution, theta_resolution):
	# Your implemention

    rows = InputImage.shape[0]
    cols = InputImage.shape[1]

    # initialize accumulator matrix
    theta_bins = int(180/theta_resolution)
    thetas = np.zeros((theta_bins,))
    for bin in range(theta_bins):
        thetas[bin] = (bin*theta_resolution)*(np.pi/180.)

    rho_max = np.sqrt(rows**2 + cols**2)
    rho_bins = int(2*rho_max/rho_resolution)
    rhos = np.zeros((rho_bins,))
    for bin in range(rho_bins):
        rhos[bin] = bin*rho_resolution-rho_max
    H = np.zeros((theta_bins, rho_bins))

    # populate H
    for y in range(rows):
        for x in range(cols):
            if InputImage[y,x] == 1:
                for bin in range(H.shape[0]):
                    theta = thetas[bin]
                    rho = x*np.cos(theta) + y*np.sin(theta)
                    rho_bin = (np.abs(rhos - rho)).argmin()
                    H[bin,rho_bin] += 1

    return H, rhos, thetas
