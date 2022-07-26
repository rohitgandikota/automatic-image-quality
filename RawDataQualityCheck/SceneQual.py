#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:27:15 2020

@author: Rohit Gandikota (NR02440)
"""
from __future__ import division
import os
from qualutils import detect_vertical_striping, detect_horizontal_striping, detect_block_loss_from_vinfo, detect_line_loss, detect_block_loss_from_hinfo, visual_block_vinfo, visual_block_hinfo, visual_striping, visual_vinfo, visual_hinfo, report_log, histogram_statistics, snr, saturated_pixels,WriteQAMeta,histogram_spread_percentage
import time
import datetime
import numpy as np
import argparse
import matplotlib.pyplot as plt

def SceneQual(sceneno,inpath, auxlen, linelen, start, scans, bytecount, display, outpath,bitperpix):    
    scenequal = {}
    AUX_LEN = auxlen
    CCD_COUNT = linelen
    BYTE_COUNT=bytecount
    LINE_BYTES=CCD_COUNT*BYTE_COUNT
    data_path=inpath
    inpath = inpath+'_'+str(sceneno)
    report_log(inpath, 'python SceneQual.py -f '+str(data_path)+' -x ' + str(sceneno)+' -a ' + str(auxlen)+ ' -l ' + str(linelen)+' -b '+ str(bytecount)+' -d '+ str(display)+' -s '+str(start)+' -n '+str(scans))
    if not (os.path.exists(data_path)):
        time.sleep(1)
        print('File Currently doesnt not exist in the specified path: '+ data_path)
        report_log(inpath, 'File Currently doesnt not exist in the specified path: ' + str(data_path))
        return 0
    start_point = start*(LINE_BYTES+AUX_LEN)
    file1_size = 0
    status = -1 
    while True:
        try:
            f2 = os.stat(data_path)
            file2_size = f2.st_size
            comp = file2_size - file1_size
            min_lines = ((comp//(LINE_BYTES+AUX_LEN)))
            if scans > 0:
                display_len = scans
            else:
                display_len = file2_size // (LINE_BYTES+AUX_LEN)
            #print(file1_size, file2_size, comp
            if min_lines > 0:
                status = 0 
                print('Reading data ....')
                report_log(inpath, 'Reading data from input '+ str(data_path))
                file= open(data_path,"rb")
                band= []
                file.seek(start_point)
                end_point = start_point + display_len * (LINE_BYTES+AUX_LEN)
                for line in range(display_len):
    
                    # - EACH LINE CONSISTS OF BYTE_COUNT*CCD_COUNT BYTES
                    # - DETERMINE THE NUMBER OF LINES = FILE_SIZE(bytes)/12000
                    # - IF NOT EXACTLY DIVISIBLE TAKE FLOOR. ( RETURN A WARNING IF POSSIBLE )
                    file.seek(AUX_LEN,1)
                    x=np.fromfile(file,np.uint16,CCD_COUNT)
                    band.append(x)
                    
                start_point = end_point
                file1_size = file1_size + display_len  * (LINE_BYTES+AUX_LEN)
                report_log(inpath, 'File size: '+ str(np.shape(np.array(band))))
                band = np.array(band)
                print('Reading of file done !')
                report_log(inpath, 'Reading of file Done !!!')
                if display == 1:
                    print('Writing the Original Image')
                    plt.imsave(inpath+'_Original.png', band,cmap='gray')
                ################################################################################## RADIOMETRY ANALYSIS
                saturated_pixel_percentage = saturated_pixels(band,bitperpix)
                report_log(inpath, 'Saturated pixel percentage calculated: '+str(saturated_pixel_percentage)+'%')
                histogram_spread = histogram_spread_percentage(band)
                report_log(inpath, 'Histogram spread percentage calculated: '+str(histogram_spread)+'%')
                data_mean, data_std, data_skew, data_kurtosis, statement = histogram_statistics(band)
                report_log(inpath, statement)
                snr_low, snr_high = snr(band)
                report_log(inpath, 'SNR calculated for low and high DN respectively: '+str(snr_low)+'dB and '+str(snr_high)+'dB')
                ################################################################################## STRIPING AND LINE LOSS
                band = np.expand_dims(band,axis=-1)
                print('Detecting vertical and Horizontal striping')
                report_log(inpath, 'Detecting vertical and Horizontal striping')
                foundVargs = detect_vertical_striping(band,enable_display=False)
                foundHargs = detect_horizontal_striping(band,enable_display=False)
                report_log(inpath, 'Vertical and Horizontal striping are as follows: '+ str(foundVargs.shape[-1])+' and '+str(foundHargs.shape[-1]))
                if display == 1:
                    print('Writing Striping Results')
                    report_log(inpath, 'Writing Striping Results')
                    visual_striping(foundVargs, foundHargs, np.shape(band), inpath)
                print('Detecting line losses')
                report_log(inpath, 'Detecting vertical and Horizontal Line Losses')
                vinfo, hinfo = detect_line_loss(band)
                if display == 1:
                    print('Writing line loss results')
                    report_log(inpath, 'Writing vertical and Horizontal Line Losses')
                    if len(vinfo)>0:
                        visual_vinfo(vinfo, band, inpath)
                    if len(hinfo)>0:
                        visual_hinfo(hinfo, band, inpath)
                report_log(inpath, 'Detecting Line Losses Done vertical and horizontal as follows: '+  str(len(vinfo))+' and '+str(len(hinfo)))
                
                ###################################################################################### SAVING AND WRITING RESULTS
                scenequal['ID'] = inpath.split('/')[-1]
                scenequal['number_of_vertical_strips'] = (foundVargs.shape[-1])
                scenequal['number_of_horizontal_strips'] = (foundHargs.shape[-1])
                scenequal['vertical_strips'] = foundVargs[0,:].tolist()
                scenequal['horizontal_strips'] = foundHargs[0,:].tolist()
                scenequal['vertical_losses'] = vinfo.tolist()
                scenequal['horizontal_losses'] = hinfo.tolist()
                scenequal['snr_ldn']=snr_low
                scenequal['snr_hdn']=snr_high
                scenequal['percentage_saturated_pixels']= saturated_pixel_percentage
                scenequal['percentage_histogram_spread']= histogram_spread
                scenequal['mean']=data_mean
                scenequal['std']=data_std
                scenequal['skew']=data_skew
                scenequal['kurtosis']=data_kurtosis
                print('Writing Scene Quality Results')
                report_log(inpath, 'Writing Scene Quality Results')
                with open(str(inpath)+ '_SceneQual.txt','w') as fp:
                    for key in scenequal.keys():
                        fp.writelines(str(key)+': '+ str(scenequal[key])+'\n')

                print('Scene Quality indicators are written to the path: ' + str(inpath)+ '_SceneQual.txt')
                report_log(inpath, 'Scene Quality indicators are written to the path: '+str(inpath) +'_SceneQual.txt')
                WriteQAMeta(scenequal, inpath)
                return 1
            else:
                if status == 0:
                    t1 = datetime.datetime.now()
                    status = 1
                elif status == -1:
                    pass
                else:
                    diff = datetime.datetime.now() - t1
                    if diff.seconds > 3:
                        #plt.imshow(labels_test,cmap='gray')
                        #plt.show()
                        print('Could not find the specified path')
                        return 0
                 
                pass
        except Exception as e:
            print('Code exited with error: '+ str(e))
            report_log(inpath, 'Code exited with error: ' + str(e))
            return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='SceneQual', description='Computes Scene Quality using Machine learning and Image processing Algorithms given vdd or subsample files.')
    parser.add_argument('-x,--sceneno', dest='sceneno',type=int, help= 'Scene Number')
    parser.add_argument('-f,--file', dest='inpath', type=str, help = 'Give the full file location for the file for which SceneQual is to be checked')
    parser.add_argument('-a,--auxlen', dest='auxlen',type=int, help= 'Aux length of file')
    parser.add_argument('-l,--linelen', dest='linelen',type=int, help= 'Number of pixels per line in the product')
    parser.add_argument('-b,--bytecount', dest='bytecount',type=int, help= 'Number of bytes per pixel')
    parser.add_argument('-d,--display', dest='display',type=int, default = 0, help= 'Give 1 if display of results required')
    parser.add_argument('-s,--start', dest='start',type=int, default = 0, help= 'Start line of the desired scene')
    parser.add_argument('-n,--noscans', dest='scans',type=int, default = -99, help= 'Number of scans in the desired scene')
    parser.add_argument('-o, --output', dest='outpath', type=str, default=os.getcwd(), help = 'Give the full file location for the text file to be written')
    parser.add_argument('-p, --bitsperpix', dest='bitperpix', type=int, default=11, help = 'Bits per pixel')
    args = parser.parse_args()
    
    SceneQual(args.sceneno,args.inpath, args.auxlen, args.linelen, args.start, args.scans, args.bytecount, args.display, args.outpath,args.bitperpix)
#    print(args)
#    inpath, auxlen, linelen, bytecount, outpath = args
    
###################################################################################################################### TESTCASES
#    inpath = '/maintenance/rohit_tmp/subpab1f_f_ai000859_000854_SAN_c03.22jan2020'
#    auxlen = 200
#    linelen= 2048   
#    bytecount = 2
#    display = 1
#    start=0
#    scans = -99
#    outpath = '/home/user/scenequal.txt'
#    SceneQual(inpath, auxlen, linelen, start, scans, bytecount, display, outpath)
    
#    
#    inpath = '/maintenance/rohit_tmp/Test/submxb2f_f_ai000867_000867_SAN_c03.23jan2020'
#    auxlen = 200
#    linelen= 2048   
#    bytecount = 2
#    display = 1
#    start= 100
#    scans = 1100
#    outpath = '/home/user/scenequal.txt'
#    SceneQual(inpath, auxlen, linelen, start, scans, bytecount, display, outpath)
##    
#    inpath = '/maintenance/rohit_tmp/rbrf001025_001021_002_ANT_pas_c03.02feb2020'
#    auxlen = 0
#    linelen= 2048   
#    bytecount = 2
#    display = 1
#    start = 0
#    scans = -99
#    outpath = '/home/user/scenequal.txt'
#    SceneQual(inpath, auxlen, linelen, start, scans, bytecount, display, outpath)
    
    
#    inpath = '/maintenance/WPDSD_IMP/.rohit_tmp/Test_New/data/submxb1f_f_ai000867_000867_SAN_c03.23jan2020'
#    sceneno=5
#    auxlen = 200
#    linelen= 2048   
#    bytecount = 2
#    display = 1
#    outpath = '/home/user/scenequal.txt'
#    start = 0
#    scans = 1000
#    bitperpix=11
#    SceneQual(sceneno,inpath, auxlen, linelen, start, scans, bytecount, display, outpath,bitperpix)
#    
#    
#    plt.imshow(band[:,:,0], cmap='gray')
#    plt.imsave('testread.png',band[:,:,0],cmap='gray')
#    python SceneQual.py -f '/maintenance/rohit_tmp/subpab1f_f_ai000859_000854_SAN_c03.22jan2020' -a 200 -l 2048 -b 2
