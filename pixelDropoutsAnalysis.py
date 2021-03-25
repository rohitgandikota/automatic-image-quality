#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 23:13:30 2021

@author: Rohit Gandikota
"""
import gdal
import numpy as np 
import os 
import glob
import matplotlib.pyplot as plt
file = '/home/rohit/wrkspc/data/R2_Bad/211361111/BAND2.tif'
band = gdal.Open(file).ReadAsArray()

hist,bins = np.histogram(band[band>0].flatten(),bins=255,range=(band.min(),band.max()))
plt.plot(hist)
wow = band[band>50]
len(wow)

mean = np.mean(band)
std = np.std(band)
lossy_patch = band[2400:2800,7500:8000]
hist,bins = np.histogram(lossy_patch[lossy_patch>0].flatten(),bins=255,range=(lossy_patch.min(),lossy_patch.max()))
plt.plot(hist)
uniform_patch=band[600:7000,7500:8000]
hist,bins = np.histogram(uniform_patch.flatten(),bins=255,range=(uniform_patch.min(),uniform_patch.max()))
plt.plot(hist)
#%% Trial on test image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

im = 'D:\\Projects\\DLIP\\SceneQual\\test_imgs\\airplane.png'
im = plt.imread(im)
im_noise = im + 0.2*np.random.randn(*im.shape)

im_med = ndimage.median_filter(im_noise, 3)

plt.figure(figsize=(16, 5))

plt.subplot(141)
plt.imshow(im, interpolation='nearest')
plt.axis('off')
plt.title('Original image', fontsize=20)
plt.subplot(142)
plt.imshow(im_noise, interpolation='nearest', vmin=0, vmax=5)
plt.axis('off')
plt.title('Noisy image', fontsize=20)
plt.subplot(143)
plt.imshow(im_med, interpolation='nearest', vmin=0, vmax=5)
plt.axis('off')
plt.title('Median filter', fontsize=20)
plt.subplot(144)
plt.imshow(np.abs(im - im_med), cmap=plt.cm.hot, interpolation='nearest')
plt.axis('off')
plt.title('Error', fontsize=20)


plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,
                    right=1)

plt.show()
#%% Trial on satellite image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

file = 'D:\\Projects\\DLIP\\RadQA\\211361111\\BAND2.tif'
band = gdal.Open(file).ReadAsArray()

im_med = ndimage.median_filter(band, 3)

plt.figure(figsize=(16, 5))

plt.subplot(131)
plt.imshow(band, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title('Original image', fontsize=20)
plt.subplot(132)
plt.imshow(im_med, cmap='gray', interpolation='nearest', vmin=0, vmax=5)
plt.axis('off')
plt.title('Median filter', fontsize=20)
plt.subplot(133)
plt.imshow(np.abs(band - im_med), cmap=plt.cm.hot, interpolation='nearest')
plt.axis('off')
plt.title('Error', fontsize=20)


plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,
                    right=1)

plt.show()
mean = np.mean(band)
std = np.std(band)
lossy_patch = band[2400:2800,7500:8000]
hist,bins = np.histogram(lossy_patch[lossy_patch>0].flatten(),bins=255,range=(lossy_patch.min(),lossy_patch.max()))
plt.plot(hist)
uniform_patch=band[600:7000,7500:8000]
hist,bins = np.histogram(uniform_patch.flatten(),bins=255,range=(uniform_patch.min(),uniform_patch.max()))
plt.plot(hist)
#%% Trial on test image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

im = 'D:\\Projects\\DLIP\\SceneQual\\test_imgs\\airplane.png'
im = plt.imread(im)
im_noise = im + 0.2*np.random.randn(*im.shape)

im_med = ndimage.median_filter(im_noise, 3)

plt.figure(figsize=(16, 5))

plt.subplot(141)
plt.imshow(im, interpolation='nearest')
plt.axis('off')
plt.title('Original image', fontsize=20)
plt.subplot(142)
plt.imshow(im_noise, interpolation='nearest', vmin=0, vmax=5)
plt.axis('off')
plt.title('Noisy image', fontsize=20)
plt.subplot(143)
plt.imshow(im_med, interpolation='nearest', vmin=0, vmax=5)
plt.axis('off')
plt.title('Median filter', fontsize=20)
plt.subplot(144)
plt.imshow(np.abs(im - im_med), cmap=plt.cm.hot, interpolation='nearest')
plt.axis('off')
plt.title('Error', fontsize=20)


plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,
                    right=1)

plt.show()
#%% Trial on satellite image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

file = 'D:\\Projects\\DLIP\\RadQA\\211361111\\BAND2.tif'
band = gdal.Open(file).ReadAsArray()

im_med = ndimage.median_filter(band, 3)

plt.figure(figsize=(16, 5))

plt.subplot(131)
plt.imshow(band, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title('Original image', fontsize=20)
plt.subplot(132)
plt.imshow(im_med, cmap='gray', interpolation='nearest', vmin=0, vmax=5)
plt.axis('off')
plt.title('Median filter', fontsize=20)
plt.subplot(133)
plt.imshow(np.abs(band - im_med), cmap=plt.cm.hot, interpolation='nearest')
plt.axis('off')
plt.title('Error', fontsize=20)


plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,
                    right=1)

plt.show()

#%% Line detection
import cv2
def lineDetection(gray):
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    
    low_threshold = 50
    high_threshold = 100
    slice1Copy = np.uint8(blur_gray)
    edges = cv2.Canny(slice1Copy, low_threshold, high_threshold)
    
    
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 1000  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(band) * 0  # creating a blank to draw lines on
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    return line_image
line_image = lineDetection(band.copy())
plt.imshow(line_image,cmap='gray')
hist,bins = np.histogram(line_image.flatten(),bins=255,range=(line_image.min(),line_image.max()))
plt.plot(hist)

#%% Pixel Droputs 
median_blur= cv2.medianBlur(band.copy(), 3)

difference = (band-median_blur)
plt.imshow(median_blur)

plt.imshow(difference)
difference[difference!=difference.max()]  = 0

plt.imshow(difference)
#%% BRISQUE image quality score
import imquality.brisque as brisque
qual_score = brisque.score(band)
rgb = np.stack([band,band,band])
np.shape(rgb)
