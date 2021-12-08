# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 10:28:26 2021

@author: Rohit Gandikota
"""
import pandas as pd
import gdal 
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
from skimage.filters import threshold_otsu

def checkImbalanceV(band):
    
    def getDisplay(band):
        rad_im = []
        for line in band.T:
            window_size = len(line)//10
            numbers_series = pd.Series(line)
            windows = numbers_series.rolling(window_size)
            moving_averages = windows.mean()
            moving_averages_list = moving_averages.tolist()
            s = moving_averages_list[window_size - 1:]
            rad_im.append(s)
        rad_im = np.array(rad_im)
        thres = threshold_otsu(rad_im)
        rad_im[rad_im<thres] = 0
        rad_im[rad_im!=0] = 1 
        return np.array(rad_im).T
    
    
    means = band.mean(axis=1)
    means_series = pd.Series(means)
    diff = pd.Series(means_series.values[1:] - means_series.values[:-1], index=means_series.index[:-1]).abs()
#    plt.plot(diff)
    diff_in_fields =  np.mean(diff[:np.argmax(diff)]) - np.mean(diff[np.argmax(diff):])
        
    if diff_in_fields > 0:
        per_change = abs(((np.mean(diff[:np.argmax(diff)]) - np.mean(diff[np.argmax(diff):]))/np.mean(diff[:np.argmax(diff)])))
    else:
        per_change = abs(((np.mean(diff[:np.argmax(diff)]) - np.mean(diff[np.argmax(diff):]))/np.mean(diff[np.argmax(diff):])))
    
    if per_change > 0.1 : 
        diff_std =  np.std(means[:np.argmax(diff)]) - np.std(means[np.argmax(diff):])
        if diff_std > 0:
            per_change = abs(((np.std(means[:np.argmax(diff)]) - np.std(means[np.argmax(diff):]))/np.std(means[:np.argmax(diff)])))
        else:
            per_change = abs(((np.std(means[:np.argmax(diff)]) - np.std(means[np.argmax(diff):]))/np.std(means[np.argmax(diff):])))
        if per_change < 50:
            rad_im = getDisplay(band)
            return True, rad_im
    else:
        return False, []
    

def checkImbalanceH(band):
    
    def getDisplay(band):
        rad_im = []
        for line in band:
            window_size = len(line)//10
            numbers_series = pd.Series(line)
            windows = numbers_series.rolling(window_size)
            moving_averages = windows.mean()
            moving_averages_list = moving_averages.tolist()
            s = moving_averages_list[window_size - 1:]
            rad_im.append(s)
        rad_im = np.array(rad_im)
        thres = threshold_otsu(rad_im)
        rad_im[rad_im<thres] = 0
        rad_im[rad_im!=0] = 1 
        return np.array(rad_im)
    
    
    means = band.mean(axis=0)
    means_series = pd.Series(means)
    diff = pd.Series(means_series.values[1:] - means_series.values[:-1], index=means_series.index[:-1]).abs()
#    plt.plot(diff)
    diff_in_fields =  np.mean(diff[:np.argmax(diff)]) - np.mean(diff[np.argmax(diff):])
        
    if diff_in_fields > 0:
        per_change = abs(((np.mean(diff[:np.argmax(diff)]) - np.mean(diff[np.argmax(diff):]))/np.mean(diff[:np.argmax(diff)])))
    else:
        per_change = abs(((np.mean(diff[:np.argmax(diff)]) - np.mean(diff[np.argmax(diff):]))/np.mean(diff[np.argmax(diff):])))
    
    if per_change > 0.1 : 
        diff_std =  np.std(means[:np.argmax(diff)]) - np.std(means[np.argmax(diff):])
        if diff_std > 0:
            per_change = abs(((np.std(means[:np.argmax(diff)]) - np.std(means[np.argmax(diff):]))/np.std(means[:np.argmax(diff)])))
        else:
            per_change = abs(((np.std(means[:np.argmax(diff)]) - np.std(means[np.argmax(diff):]))/np.std(means[np.argmax(diff):])))
        if per_change < 50:
            rad_im = getDisplay(band)
            return True, rad_im
    else:
        return False, []


band = gdal.Open('D:\\Projects\\PQC\\image_HH.tif').ReadAsArray()
band = gdal.Open('D:\\Projects\\PQC\\RadQA\\213626921\\BAND2_RPC.tif').ReadAsArray()
flag, rad_im = checkImbalanceH(band)
#flag, rad_im = checkImbalanceV(band)
plt.imshow(std_clip(band),cmap='gray')
plt.imshow(rad_im,cmap='gray')
