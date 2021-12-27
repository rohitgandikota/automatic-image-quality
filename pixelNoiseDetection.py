#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:54:46 2021

@author: Rohit Gandikota
"""
import numpy as np
from scipy import ndimage
import gdal
import matplotlib.pyplot as plt

def pixelNoise(band,display = 1):
    '''
    Function to detect random pixel noise in an image. Can be used to detect 
    random pixel errors in satellite imagery, especially RS2 and R2A satellites.

    Parameters
    ----------
    band : ndarray
        The band data that is being evaluated.
    display : boolean, optional
        Flag used to return the masking image. The default is 1.

    Returns
    -------
    int
        Number of total pixels noises detected in the satellite image.
    ndarray, conditional on "display" flag
        The array representing the pixel locations. Same size as band. 255 if line; else 0.

    '''
    im_med = ndimage.median_filter(band,3)
    error = np.abs(band - im_med)
    
    mean = np.mean(error)
    std = np.std(error)
    
    error[error<mean+2*std] = 0
    error[error>0] = 1
    if display == 1:
        return len(error[error==1].flatten()),error
    else:
        return len(error[error==1].flatten())
    
def detect_sp(patch, w):
    """
    Algorithm 1 (Basic) for detection of salt and pepper noise
    """
    C = 0
    centre  = int(patch.shape[0]/2)
    pix = patch[centre,centre]
    T = 0.5
#    T = w*w - (2*w)
    patch = np.array(patch)
    for neigh in np.nditer(patch):
        G = abs(pix - (neigh+1))/(neigh+1)
        if G > T or G ==T:
            C += 1
    if C > 2*w*(2*w+1) and (pix == 255 or pix == 0):
        return 1
    else:
        return 0
        
def detect_sp_advanced(image):
    """
    An advanced adaptive window sized s&p noise detection using Algorithm 1
    """
    windows = [3]
    detected = np.zeros(image.shape)
    for window in windows:
        for k in range(int(image.shape[2])):
            print('Detecting Channel:' + str(k))
            for i in range(int(image.shape[0])):
                for j in range(int(image.shape[1])):
                    patch = image[(i-window):(i+window+1), (j-window):(j+window+1),k]
                    try:
                        med = np.median(patch)
                        maxi = np.max(patch)
                        mini = np.min(patch)
                        if med<maxi and med>mini:
                            if image[i,j,k] < maxi and image[i,j,k]>mini:
                                pass
                            else:
                                detected[i,j,k] =  detect_sp(patch, window)
                    except:
                        pass
                    
    return detected


def detect_sp_new(image):
    """
    An advanced adaptive window sized s&p noise detection using Algorithm 1
    """
    windows = [3]
    detected = np.zeros(image.shape)
    for window in windows:
        for k in range(int(image.shape[2])):
            print('Detecting Channel:' + str(k))
            for i in range(int(image.shape[0])):
                for j in range(int(image.shape[1])):
                    patch = image[(i-window):(i+window+1), (j-window):(j+window+1),k]
                    try:
                        detected[i,j,k] =  detect_sp(patch, window)
                    except:
                        pass
                    
    return detected

if __name__=='__main__':

    file = 'BAND.tif'

    
    band = gdal.Open(file).ReadAsArray()
    im_med = ndimage.median_filter(band,3)
    badPixelCount, error = pixelNoise(band,display = 1)
    plt.figure(figsize=(16, 5))
    
    plt.subplot(131)
    plt.imshow(band, cmap='gray')
    plt.axis('off')
    plt.title('Original image', fontsize=20)
    plt.subplot(132)
    plt.imshow(im_med, cmap='gray')
    plt.axis('off')
    plt.title('Median filter', fontsize=20)
    plt.subplot(133)
    plt.imshow(np.abs(band - im_med), cmap='gray')
    plt.axis('off')
    plt.title('Error', fontsize=20)
    
    
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,
                        right=1)
    
    plt.show()
