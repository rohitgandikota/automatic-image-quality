# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:12:46 2019

@author: Rohit Gandikota (NR02440)
"""
import numpy as np
import os
#import cv2
import numpy as np
import os
#import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random
from scipy.stats import wasserstein_distance
from scipy.signal import find_peaks
from argparse import ArgumentParser
from scipy.stats import kurtosis,skew
from scipy.ndimage.filters import convolve
import datetime, sys 

# SNR code 
def signaltonoise(a):
    a = np.asanyarray(a)
    m = a.mean(None)
    sd = a.std(axis=None, ddof=0)
#    print('Mean =' + str(m) + ' Std= ' + str(sd))
    if sd == 0:
        return 0
    else :
        return np.float(m/sd)
def snr(band, ksize=5):
    
    kernel = np.ones((ksize,ksize)) / ksize*ksize
    
    mean_grid = convolve(band, kernel, mode='reflect')
    
    argmax = np.unravel_index(np.argmax(mean_grid),np.shape(mean_grid))
    mean_grid[mean_grid==0]=np.max(mean_grid)
    argmin = np.unravel_index(np.argmin(mean_grid),np.shape(mean_grid))
    
    patch = band[argmin[0]-(ksize//2):argmin[0]+(ksize//2), argmin[1]-(ksize//2):argmin[1]+(ksize//2)]
    
    min_snr = signaltonoise(patch)
    
    patch = band[argmax[0]-(ksize//2):argmax[0]+(ksize//2), argmax[1]-(ksize//2):argmax[1]+(ksize//2)]
    
    max_snr = signaltonoise(patch)

    return min_snr, max_snr
# Saturated pixels
def saturated_pixels(band,bitperpix):
    shape = (np.shape(band))
    total_pix = np.float(shape[0]*shape[1])
    sat_pix = np.float(len(band[band==(2**bitperpix)-1]))
    return round(sat_pix/total_pix,3)
def histogram_spread_percentage(band):
    hist,bins = np.histogram(band, bins=255)
    
    saturated = np.float(hist[0]+hist[1] + hist[-1]+hist[-2])
    length= np.float(len(band.flatten()))
    saturated_pixel_percentage = np.float(saturated / length)* 100
    return saturated_pixel_percentage
# Histogram Statistics
def histogram_statistics(band):
    ################################################################################# Mean, Standard Deviation, Kurtosis and Skew
    data_mean = np.mean(band)
    data_std = np.std(band)
    hist = np.histogram(band,bins=255)[0][1:]
    data_skew = skew(hist)
    data_kurtosis = kurtosis(hist)
    if data_skew > 0:
        statement = 'From the values, the histogram is skewed towards right and '
    else:
        statement = 'From the values, the histogram is skewed towards left and '
    if data_kurtosis >4:
        statement += 'histogram is peaky'
    else:
        statement += 'histogram is close to normal distribution'
    return data_mean, data_std, data_skew, data_kurtosis, statement
    
# Detection Algorithms
def detect_line_loss(noisy):
    """
    TODO
    What if two discontinuous losses
    Noisy image -> Vertical and Horizontal line loss locations
    
    Image -> ((row, band, startpixel, number of pixels), (column, band, startpixel, number of pixels))
    """
    shape = noisy.shape
    vinfo = []
    hinfo = []
    for i in range(shape[0]):
        for j in range(shape[-1]):
            diff1 = noisy[i,:,j] - np.roll(noisy[i,:,j],-1)
            args = np.argwhere(diff1 == 0 )
            args_rolled = np.roll(args, -1, axis = 0)
            diff = args_rolled - args
            lines = np.argwhere(diff!=1)
            for k in range(len(lines)):
                if k == 0:
                    start = args[0][0]
                    end = args[lines[k][0]][0]
                else:
                    start = args[lines[k-1][0]+1][0]
                    end = args[lines[k][0]][0]
                if end - start > 4:
                        hinfo.append([j, i, start, end])
                
    for i in range(shape[1]):
        for j in range(shape[-1]):
            diff1 = noisy[:,i,j] - np.roll(noisy[:,i,j],-1)
            args = np.argwhere(diff1 == 0 )
            args_rolled = np.roll(args, -1, axis = 0)
            diff = args_rolled - args
            lines = np.argwhere(diff!=1)
            for k in range(len(lines)):
                if k == 0:
                    start = args[0][0]
                    end = args[lines[k][0]][0]
                else:
                    start = args[lines[k-1][0]+1][0]
                    end = args[lines[k][0]][0]
                if end - start > 4:
                        vinfo.append([j, i, start, end])
    return np.array(vinfo), np.array(hinfo)


def detect_vertical_striping(noisy_img,enable_display=False):
    """
    Noisy Image -> (List of all args, list of each band args)
    Takes the noisy image and returns the array of all possible lines in error along with error lines per band
    """
    noisy = noisy_img.copy()
    output = []
    for j in range(noisy.shape[-1]):
        distances = []
        for i in range(1,noisy.shape[1]-1):
            hist1, bin1 = np.histogram(noisy[:,i-1,j],100)
            hist2, bin1 = np.histogram(noisy[:,i,j],bins = bin1)
            hist3, bin1 = np.histogram(noisy[:,i+1,j],bins = bin1)
            dist = wasserstein_distance(hist1, hist2) + wasserstein_distance(hist2, hist3)
            distances.append(dist)
        if enable_display:
            plt.plot(distances)
        boom = find_peaks(distances, height=((.5/100)*noisy.shape[0]))
        output.append(boom[0]+1)
#    A = list(set(output[0])|set(output[1])|set(output[2]))
    return np.array(output) 

def detect_horizontal_striping(noisy_img,enable_display=False):
    """
    Noisy Image -> (List of all args, list of each band args)
    Takes the noisy image and returns the array of all possible lines in error along with error lines per band
    """
    noisy = noisy_img.copy()
    output = []
    for j in range(noisy.shape[-1]):
        distances = []
        for i in range(1,noisy.shape[0]-1):
            hist1, bin1 = np.histogram(noisy[i-1,:,j],100)
            hist2, bin1 = np.histogram(noisy[i,:,j],bins = bin1)
            hist3, bin1 = np.histogram(noisy[i+1,:,j],bins = bin1)
            dist = wasserstein_distance(hist1, hist2) + wasserstein_distance(hist2, hist3)
            distances.append(dist)
        if enable_display:
            plt.plot(distances)
        boom = find_peaks(distances, height=((.5/100)*noisy.shape[1]))
        output.append(boom[0]+1)
#    A = list(set(output[0])|set(output[1])|set(output[2]))
    return np.array(output) 

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
#            print('Detecting Channel:' + str(k))
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
#            print('Detecting Channel:' + str(k))
            for i in range(int(image.shape[0])):
                for j in range(int(image.shape[1])):
                    patch = image[(i-window):(i+window+1), (j-window):(j+window+1),k]
                    try:
                        detected[i,j,k] =  detect_sp(patch, window)
                    except:
                        pass
                    
    return detected

def visual_vinfo(vinfo, image, inpath):
    output = np.zeros(image.shape)
    hello = vinfo[vinfo[:,0] == 0]
    for hel in hello:
        output[hel[2]:hel[3], hel[1],0] = 255
    plt.imsave(inpath+'_vinfo.png',output[:,:,0],cmap='gray')
    
def visual_hinfo(hinfo, image, inpath):
    output = np.zeros(image.shape)
    hello = hinfo[hinfo[:,0] == 0]
    for hel in hello:
        output[hel[1], hel[2]:hel[3],0] = 255
    plt.imsave(inpath+'_hinfo.png',output[:,:,0],cmap='gray')
    
    

def visual_striping(foundVargs, foundHargs, shape, inpath):
    V_img = np.zeros(shape)
    H_img = np.zeros(shape)
    V_img[:, foundVargs, 0] = 255
    H_img[foundHargs, :, 0] = 255
#    for i in range(len(foundVargs)//2):    
#        V_img[:,foundVargs[2*i]:foundVargs[2*i+1],:] = 1
#    for i in range(len(foundHargs)//2):    
#        H_img[foundHargs[2*i]:foundHargs[2*i+1],:,:] = 1
    plt.imsave(inpath+'_VerticalStriping.png',V_img[:,:,0],cmap='gray')
    plt.imsave(inpath+'_HorizontalStriping.png',H_img[:,:,0],cmap='gray')
def report_log(inpath, string):
    browse_id = inpath.split('/')[-1]
    designer_path=checkPath(browse_id)
    if os.path.exists(designer_path+'/'+browse_id+'_log.txt'):
        ftype = 'a'
    else:
        ftype = 'w'
    with open(designer_path+'/'+browse_id+'_log.txt',ftype) as fp:
        fp.write('\n '+str(datetime.datetime.now()) + ':: \t '+str(string) +' \n')
        
    if os.path.exists(inpath+'_log.txt'):
        ftype = 'a'
    else:
        ftype = 'w'
    with open(inpath+'_log.txt',ftype) as fp:
        fp.write('\n '+str(datetime.datetime.now()) + ':: \t '+str(string) +' \n')
def checkPath(browse_id):
    dop = browse_id.split('_')[-2].split('.')[-1]
    month = dop[2:5].upper()
    year = dop[-4:]
    string = '/'+str(year)+'/'+str(month)+'/'+browse_id
    path = '/prodspace/PQC/RadQAOper/Reports'+string
    if not os.path.exists(path):
        os.makedirs(path)
    return path
def WriteQAMeta(scenequal, inpath):
    browse_id = inpath.split('/')[-1]
    
    designer_path=checkPath(browse_id)
    with open(designer_path+'/'+browse_id+'_SceneQual_META.txt','w') as fp:
        fp.writelines('BrowseID='+str(scenequal["ID"])+'\n')
        fp.writelines('Mean='+str(scenequal["mean"])+'\n')
        fp.writelines('StandardDeviation='+str(scenequal["std"])+'\n')
        fp.writelines('Skew='+str(scenequal["skew"])+'\n')
        fp.writelines('Kurtosis='+str(scenequal["kurtosis"])+'\n')
        fp.writelines('SaturatedPixelsPercentage='+str(scenequal["percentage_saturated_pixels"])+'\n')
        fp.writelines('SNRatHighDN='+str(scenequal["snr_hdn"])+'\n')
        fp.writelines('SNRatLowDN='+str(scenequal["snr_ldn"])+'\n')
        fp.writelines('VerticalStripingCount='+str(scenequal["number_of_vertical_strips"])+'\n')
        fp.writelines('HorizontalStripingCount='+str(scenequal["number_of_horizontal_strips"])+'\n')
        fp.writelines('LineLossPercentage='+str(scenequal["percentage_area_of_line_losses_in_pixels"])+'\n')
    with open(inpath+'_SceneQual_META.txt','w') as fp:
        fp.writelines('BrowseID='+str(scenequal["ID"])+'\n')
        fp.writelines('Mean='+str(scenequal["mean"])+'\n')
        fp.writelines('StandardDeviation='+str(scenequal["std"])+'\n')
        fp.writelines('Skew='+str(scenequal["skew"])+'\n')
        fp.writelines('Kurtosis='+str(scenequal["kurtosis"])+'\n')
        fp.writelines('SaturatedPixelsPercentage='+str(scenequal["percentage_saturated_pixels"])+'\n')
        fp.writelines('SNRatHighDN='+str(scenequal["snr_hdn"])+'\n')
        fp.writelines('SNRatLowDN='+str(scenequal["snr_ldn"])+'\n')
        fp.writelines('VerticalStripingCount='+str(scenequal["number_of_vertical_strips"])+'\n')
        fp.writelines('HorizontalStripingCount='+str(scenequal["number_of_horizontal_strips"])+'\n')
        fp.writelines('LineLossPercentage='+str(scenequal["percentage_area_of_line_losses_in_pixels"])+'\n')
