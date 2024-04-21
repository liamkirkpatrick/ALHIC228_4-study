#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:24:30 2024

Compute correct dip angle for ALHIC2201

@author: Liam
"""

#%% Import packages and Functions

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import math
from tqdm import tqdm
import warnings
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
import matplotlib.ticker as plticker

# import Functions
sys.path.append('../../../data/core_scripts')
from ecmclass import ecmdata

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


#%% Set Variables

# smoothing Window
window=1

# Cross Correlation Parameters

anglespace = .01         # spacing between degrees
comp_range = 0.25       # distance over which to complete correlation comparison
comp_int = 0.01         # depth interval to complete correlation comparison
interp_int = 0.00025    # spacing of line that we interpolate onto
pad_offset = 1          # padding from edge, mm


#%% Housekeeping

# set file paths
path_to_data = '../data/ECM/'


# Import Color Maps
cmap1 = plt.cm.Paired

# define legend to plot w/o duplicate labels
def legend_without_duplicate_labels(ax,location):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    if y:
        ax.legend(*zip(*unique),loc=location)
    else:
        ax.legend(*zip(*unique),loc=location)
#%% 1 - import ECM data

# data info
left_file = "cmc1-228_4-L_AC_2023-06-16-13-24.npy"
right_file = "cmc1-228_4-R_AC_2023-06-16-11-00.npy"
top_file = "cmc1-228-4_AC_2023-06-15-15-10.npy"
core_name = "CMC1-228_4-AC"

print("********************************************")
print("Running file "+core_name+" with:\n"+
      "Left File: "+left_file+"\n"+
      "Right File: "+right_file+"\n"+
      "Top File: "+top_file)
print("********************************************")


# read in data
left = ecmdata(path_to_data+left_file,1)
right = ecmdata(path_to_data+right_file,1)
top = ecmdata(path_to_data+top_file,1)


#%%  4 - Calculate dip angle on all faces

def find_angle(data,anglemin,anglemax):
    
    # to start, let's assign some variables
    depth = data.depth_smooth
    meas = data.meas_smooth
    y_smooth = data.y_smooth
    
    # set angles to cycle through
    test_angle = np.round(np.linspace(anglemin,anglemax,round((anglemax-anglemin)/anglespace+1)),4)
    
    # shorten y_vec for top faces 
    y_vec = data.y_vec
    
    # get window from edge of core for padding
    y_space = y_vec[1]-y_vec[0] # calculate spacing between runs
    angle_offset = (y_space/2) * math.tan(max(abs(anglemin),abs(anglemax)) * (2*math.pi/360))
    offset = (angle_offset + pad_offset) /1000 #offset, in m
    
    # make evenly spaced grid
    grid_min = math.ceil(min(data.depth_smooth)*100)/100
    grid_max = math.floor(max(data.depth_smooth)*100)/100
    grid = np.linspace(grid_min,
            grid_min + math.floor((grid_max-grid_min)/interp_int)*interp_int,
            math.floor((grid_max-grid_min)/interp_int)+1)
    
    # make vector of depths at which we will calculate slopes
    min_depth = min(data.depth_smooth)+offset
    max_depth = max(data.depth_smooth)-offset
    d_vec = np.linspace(min_depth,
                    min_depth + math.floor((max_depth-min_depth)/comp_int)*comp_int,
                    math.floor((max_depth-min_depth)/comp_int)+1)
    
    # Make empty numpy array to store correlation coefs
    coefs_array = np.zeros([len(y_vec)-3, len(d_vec)])
    angle_array = np.zeros([len(y_vec)-3, len(d_vec)])
    
    # Find index for each track
    track_idx= []
    for y in y_vec:
        track_idx.append(y_smooth==y)
    
    # calculate shift at each angle
    for s in test_angle:
        
        shift = ((y_space/2/1000) * math.tan(s * (2*math.pi/360)))
        print("Running Angle "+str(s)+", with shift "+str(round(shift,5))+"mm")
    
    
        # Loop through all tracks, skipping the outermost track on each side
        ycnt = 0
        for i in range(1,len(y_vec)-2):
            
            #print("Running Tracks "+str(y_vec[i])+" and "+str(y_vec[i+1]))
            
            #idx_y1 = y_smooth==y_vec[i]
            #idx_y2 = y_smooth==y_vec[i+1]
            idx_y1 = track_idx[i]
            idx_y2 = track_idx[i+1]
        
            # assign right side of pair (interpolated onto depth grid)
            arr_r = np.interp(grid,np.flip(depth[idx_y1]+shift),np.flip(meas[idx_y1]))
            y_r = y_vec[i]
            
            # assign left side of pair (interpolated onto depth grid)
            arr_l = np.interp(grid,np.flip(depth[idx_y2]-shift),np.flip(meas[idx_y2]))
            y_l = y_vec[i+1]
            
            corr_coef = []
            # Loop through all depths in d_vec
            dcnt = 0
            for d in d_vec:
                
                # find min/max of all samples within range
                dmax = d + comp_range/2
                dmin = d - comp_range/2
                
                # find all points within the grid
                idx1 = grid>=dmin
                idx2 = grid<=dmax
                idx = idx1*idx2
                
                # Run Correlation
                corr_coef,_=pearsonr(arr_r[idx],arr_l[idx])
                
                # Check if this is a stronger correlation at this depth than all
                # other runs at different slopes
                if corr_coef >= coefs_array[ycnt,dcnt]:
                    coefs_array[ycnt,dcnt] = corr_coef
                    angle_array[ycnt,dcnt] = s
                
                # increment depth step counter
                dcnt+=1
            
            # increment track step counter
            ycnt+=1
            
    # Calculate Average Dip
    dip = np.mean(angle_array)
    
    print("Dip angle is "+str(round(dip,4)))

    return(dip)


# Run Function to adjust by level
top_angle = find_angle(top,-45,-25)
left_angle = find_angle(left,5,25)
right_angle = find_angle(right,-25,-5)
side_angle = (left_angle*-1+right_angle)/2   # average of left and right

df = pd.DataFrame({'Top Angle': [top_angle],'Side Angle':[side_angle]})
df.to_csv(path_to_data+'dip_angle.csv')

