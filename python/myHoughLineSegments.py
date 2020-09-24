import glob
import os.path as osp
import numpy as np
import cv2

def drawLines(img_in, start_points, end_points):
    min_length = 5
    img_output = img_in.copy()
    for i in range(len(start_points)):
        if ((start_points[i][0]-end_points[i][0])**2+(start_points[i][1]-end_points[i][1])**2)**0.5 > min_length:
            img_output = cv2.line(img_in, start_points[i], end_points[i], (0,255,0), 1)
    return img_output

def slopeInterceptForm(nLines, peakRho, peakTheta, rhosscale, thetasscale):
    m = np.zeros((nLines,))
    b = np.zeros((nLines,))
    # calculate m and b for each (theta,rho)
    for i in range(nLines):
        theta = thetasscale[int(peakTheta[i])]
        rho = rhosscale[int(peakRho[i])]
        if theta == 0.:
            m[i] = np.inf
            b[i] = 0.
        else:
            m[i] = -1. / np.tan(theta)
            b[i] = rho / np.sin(theta)
    return m,b

def myHoughLineSegments(img_in, edgeimage, peakRho, peakTheta, rhosscale, thetasscale):
    # Your implemention
    nLines = len(peakTheta)
    rows = img_in.shape[0]
    cols = img_in.shape[1]

    m,b = slopeInterceptForm(nLines, peakRho, peakTheta, rhosscale, thetasscale)

    start_points = []
    end_points = []
    # for each line
    for i in range(nLines):
        segment_started = False
        last_point = (0,0)
        if m[i] == np.inf or m[i] == -np.inf: # special case
            if peakRho[i]>=0 and peakRho[i]<cols: # check bounds
                x = int(peakRho[i])
                for y in range(rows):
                    if edgeimage[max(y-1,0):min(y+1,rows),max(x-1,0):min(x+1,cols)].max() == 1.0: # check for local edge
                        if not segment_started:
                            start_points.append((x,y))
                            segment_started = True
                        last_point = (x,y)
                    elif segment_started:
                        end_points.append((x,y))
                        segment_started = False
        elif abs(m[i]) > 1.: # large slope - better results by iterating over y
            # for each pixel on the line x = (y - b) / m
            for y in range(rows):
                x = int((y - b[i])/m[i])
                if x>=0 and x<cols and edgeimage[max(y-1,0):min(y+1,rows),max(x-1,0):min(x+1,cols)].max() == 1.0: # check for local edge
                    if not segment_started: # start new segment
                        start_points.append((x,y))
                        segment_started = True
                    last_point = (x,y)
                elif x>=0 and x<cols and segment_started:
                    end_points.append((x,y))
                    segment_started = False
        else:
            # for each pixel on the line y = mx + b
            for x in range(cols):
                y = int(m[i]*x+b[i])
                if y>=0 and y<rows and edgeimage[max(y-1,0):min(y+1,rows),max(x-1,0):min(x+1,cols)].max() == 1.0: # check for local edge
                    if not segment_started: # start new segment
                        start_points.append((x,y))
                        segment_started = True
                    last_point = (x,y)
                elif y>=0 and y<rows and segment_started:
                    end_points.append((x,y))
                    segment_started = False
        if segment_started:
            end_points.append(last_point)
            segment_started = False
    img_output = drawLines(img_in, start_points, end_points)

    return img_output
