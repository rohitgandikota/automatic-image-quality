#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:49:30 2021

@author: Rohit Gandikota
"""
import cv2
import numpy as np
def lineDetection(band,min_line_length=10000):
    '''
    Function to detect lines in an image. Can be used to detect striping and 
    banding in any satellite image irrespective of the level of correction. 
    Parameters
    ----------
    band : ndarray
        The band data that is being evaluated.
    min_line_length : int ,optional
        The minimum pixel count criteria for a line to be detected. Default 10000

    Returns
    -------
    int
        Number of total lines detected in the satellite image.
    line_image : ndarray
        The array representing the line locations. Same size as band. 255 if line; else 0

    '''
    def auto_canny(image, sigma=0.33):
    	# compute the median of the single channel pixel intensities
    	v = np.median(image)
    	# apply automatic Canny edge detection using the computed median
    	lower = int(max(0, (1.0 - sigma) * v))
    	upper = int(min(255, (1.0 + sigma) * v))
    	edged = cv2.Canny(image, lower, upper)
    	# return the edged image
    	return edged
    kernel_size = 5
    blur_band = cv2.GaussianBlur(band,(kernel_size, kernel_size),0)
    
    # low_threshold = 50
    # high_threshold = 100
    slice1Copy = np.uint8(blur_band)
    # edges = cv2.Canny(slice1Copy, low_threshold, high_threshold)
    edges = auto_canny(slice1Copy)
    
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = min_line_length  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(band) * 0  # creating a blank to draw lines on
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    if lines!=None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    
        return len(lines), line_image
    else:
        return 0,line_image
        
if __name__ == "__main__":
    import gdal
    import matplotlib.pyplot as plt
   
    band = gdal.Open('BAND.tif').ReadAsArray()
    lineCount, line_image = lineDetection(band.copy(),min_line_length=10000)
    plt.imshow(line_image,cmap='gray')
   