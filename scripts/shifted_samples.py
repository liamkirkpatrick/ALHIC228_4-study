#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:23:39 2024

Plot locations of discrete samples overlaid on ECM (left pane)
and their values on shifted depth (right pane)

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
import matplotlib
from matplotlib.patches import Rectangle
import os
import moviepy.video.io.ImageSequenceClip
import copy
import re


# import Functions
sys.path.append('../../../data/core_scripts')
from ecmclass import ecmdata


#%% Set Variables

# smoothing Window
window=1

# Cross Correlation Parameters
anglespace = .05         # spacing between degrees
comp_range = 0.10       # distance over which to complete correlation comparison
comp_int = 0.001         # depth interval to complete correlation comparison
interp_int = 0.00025    # spacing of line that we interpolate onto
pad_offset = 1          # padding from edge, mm

#FUDGE FOR LEFT/RIGHT
# TEMPORARY - AJDUSTS L/R ECM to match top
# NEED TO ACTUALLY QUANTIFY WHERE THIS ERROR IS COMING FROM
fudge = 0.034
 
#%% Housekeeping

# set file paths
path_to_data = '../data/'
path_to_figures = '../../figures/'

# Import Color Maps
cmap1 = plt.cm.Paired

#%% import Data

#Import ECM
data = ecmdata(path_to_data+'ECM/cmc1-228-4_AC_2023-06-15-15-10.npy',10)

# Import Sample data
master = pd.read_csv(path_to_data+'master_samples.csv',header=[0],index_col=[0])

# Import angles
angles = pd.read_csv(path_to_data+'ECM/dip_angle.csv')
top_angle = -angles['Top Angle'].to_numpy().item()
side_angle = angles['Side Angle'].to_numpy().item()
#side_angle = 0

#%% Setup for plot

# to start, let's assign some variables
y_vec = data.y_vec
depth = data.depth_smooth
meas = data.meas_smooth
y_smooth = data.y_smooth
button_smooth = data.button_smooth

# set up colors
pltmin = 1.3*10**(-8)
pltmax = 2.0*10**(-8)
colors = ([0,0,1],[1,0,0])
my_cmap = matplotlib.colormaps['Spectral']
rescale = lambda k: (k-pltmin) /  (pltmax-pltmin)

# find y middle
middle = (max(y_vec)-min(y_vec))/2

# Get list of things we want to plot
excluded_words = ['Note', 'depth', 'center']
samps_to_plot = [col for col in master.columns if not any(ex_word in col for ex_word in excluded_words)]
samps_labels = ['dust' + col +'µm' if 'Vol' in col else col for col in samps_to_plot]

#%% define plot box function

def plotsamps(top,bot,l,r,ax,label,c):

    for i in range(len(top)):
        ax.plot([l,r],[top[i],top[i]],color=c,linewidth=1)
        ax.plot([l,l],[top[i],bot[i]],color=c,linewidth=1)
        ax.plot([r,r],[top[i],bot[i]],color=c,linewidth=1)
        ax.plot([l,r],[bot[i],bot[i]],color=c,linewidth=1)

def plotsamps(df,ax,sticks):

        
    for index, row in df.iterrows():
        
        prefix_index = next((i for i, prefix in enumerate(sticks) if index.startswith(prefix)), None)
        
        color = cmap1(prefix_index*2+7)
        
        # list all boxes
        TL = row['TL_dipadj']
        TR = row['TR_dipadj']
        BL = row['BL_dipadj']
        BR = row['BR_dipadj']
        l = row['left_fromcenter_mm']
        r = row['right_fromcenter_mm']
        
        ax.plot([l,r],[TL,TR],color=color,linewidth=1)
        ax.plot([l,l],[TL,BL],color=color,linewidth=1)
        ax.plot([r,r],[TR,BR],color=color,linewidth=1)
        ax.plot([l,r],[BL,BR],color=color,linewidth=1)

#%% Make ECM Plot

angle = 0
print('Plotting ECM Background - angle = '+str(round(angle,2)))
# Make plot
ECMfig0,ECMaxs0 = plt.subplots(1,2,dpi=200,figsize=(8,5))
# Axis 0 lables and  limits
ECMaxs0[0].set_ylabel('Dip-Adjusted Depth (m)')
ECMaxs0[0].set_xlabel('Distance Accross Core (mm)')
ECMaxs0[0].set_ylim([155.4,155.0])
ECMaxs0[0].set_xlim([130,-130])
# Axis 1 lables and  limits
ECMaxs0[1].set_ylabel('Dip-Adjusted Depth (m)')
ECMaxs0[1].set_ylim([155.4,155.0])
ECMaxs0[1].yaxis.tick_right()
ECMaxs0[1].yaxis.set_label_position("right")

# Subplot #1 
for j in range(len(y_vec)):

    measy = meas[y_smooth==y_vec[j]]
    depthy = depth[y_smooth==y_vec[j]]

    depth_adj = depthy - np.sin(-angle*np.pi/180) * (y_vec[j]-middle)/1000


    for i in range(len(measy)-1):
    
        ECMaxs0[0].add_patch(Rectangle((y_vec[j]-4.8-middle,depth_adj[i]),9.6,depth_adj[i+1]-depth_adj[i],facecolor=my_cmap(rescale(measy[i]))))


angle = top_angle
print('Plotting ECM Background - angle ='+str(round(angle,2)))
# Make plot
ECMfig1,ECMaxs1 = plt.subplots(1,2,dpi=200,figsize=(8,5))
# Axis 0 lables and  limits
ECMaxs1[0].set_ylabel('Dip-Adjusted Depth (m)')
ECMaxs1[0].set_xlabel('Distance Accross Core (mm)')
ECMaxs1[0].set_ylim([155.4,155.0])
ECMaxs1[0].set_xlim([130,-130])
# Axis 1 lables and  limits
ECMaxs1[1].set_ylabel('Dip-Adjusted Depth (m)')
ECMaxs1[1].set_ylim([155.4,155.0])
ECMaxs1[1].yaxis.tick_right()
ECMaxs1[1].yaxis.set_label_position("right")

# Subplot #1 
for j in range(len(y_vec)):

    measy = meas[y_smooth==y_vec[j]]
    depthy = depth[y_smooth==y_vec[j]]

    depth_adj = depthy - np.sin(-angle*np.pi/180) * (y_vec[j]-middle)/1000


    for i in range(len(measy)-1):
    
        ECMaxs1[0].add_patch(Rectangle((y_vec[j]-4.8-middle,depth_adj[i]),9.6,depth_adj[i+1]-depth_adj[i],facecolor=my_cmap(rescale(measy[i]))))



#%% loop though all elements

for col in samps_to_plot:
    
    print('Plotting '+col)
    
    # select appropriate data
    selected_columns = ['rawdepth_top_m','rawdepth_bottom_m',
                        'left_fromcenter_mm','right_fromcenter_mm',
                        'top_fromcenter_mm','bot_fromcenter_mm',col]



    
    for (angle,ECMfig) in zip([0, top_angle],[ECMfig0,ECMfig1]):
        
        df = master[selected_columns].dropna()
        ave_intopage = (df['top_fromcenter_mm']+df['bot_fromcenter_mm'])/2
        ave_crosspage = (df['left_fromcenter_mm']+df['right_fromcenter_mm'])/2
        
        # get unqiue sticks
        sticks = [re.sub(r'\d+', '', s) for s in df.index.tolist()]
        sticks = list(set(sticks))
        
        print("     Angle "+str(round(angle,2)))
        
        # compute adjusted dips (all four corners)
        df['TL_dipadj'] = (df['rawdepth_top_m'] 
                           + ave_intopage/1000 * np.sin(side_angle*np.pi/180)
                           + df['left_fromcenter_mm']/1000 
                           * np.sin(angle*np.pi/180))
        df['TR_dipadj'] = (df['rawdepth_top_m'] 
                           + ave_intopage/1000 * np.sin(side_angle*np.pi/180)
                           + df['right_fromcenter_mm']/1000 
                           * np.sin(angle*np.pi/180))
        df['BL_dipadj'] = (df['rawdepth_bottom_m'] 
                           + ave_intopage/1000 * np.sin(side_angle*np.pi/180)
                           + df['left_fromcenter_mm']/1000 
                           * np.sin(angle*np.pi/180))
        df['BR_dipadj'] = (df['rawdepth_bottom_m'] 
                           + ave_intopage/1000 * np.sin(side_angle*np.pi/180)
                           + df['right_fromcenter_mm']/1000 
                           * np.sin(angle*np.pi/180))
        df['topdepth_dipadj'] = (df['rawdepth_top_m'] 
                           + ave_intopage/1000 * np.sin(side_angle*np.pi/180)
                           + ave_crosspage/1000 * np.sin(angle*np.pi/180))
        df['botdepth_dipadj'] = (df['rawdepth_bottom_m'] 
                           + ave_intopage/1000 * np.sin(side_angle*np.pi/180)
                           + ave_crosspage/1000 * np.sin(angle*np.pi/180))

        
        # Make plot
        fig = copy.deepcopy(ECMfig)
        axs = fig.axes  # Get the copy of the axes
        axs[1].set_ylabel('Dip-Adjusted Depth (m)')
        
        # left pannel
        idxcnt = 0
        unique_indexes = df.index.unique()
        for idx in unique_indexes:
            df_subset = df[df.index.to_series().str.contains(idx)]
            plotsamps(df_subset,axs[0],sticks)
            
            idxcnt+=1
            
        # right pannel
        for index, row in df.iterrows():
            
            prefix_index = next((i for i, prefix in enumerate(sticks) if index.startswith(prefix)), None)
            
            
            color = cmap1(prefix_index*2+7)
            
            axs[1].plot([row[col],row[col]],
                        [row['topdepth_dipadj'],row['botdepth_dipadj']],
                        color=color)
        cnt = 0
        for s in sticks:
            index = list(df.index)
            df_subset = df[df.index.to_series().str.contains(s)]
            axs[1].plot(df_subset[col],(df_subset['topdepth_dipadj']+df_subset['botdepth_dipadj'])/2,'-',color=cmap1(cnt*2+7))
            cnt+=1
         
        # plot housekeeping
        axs[1].set_ylabel(col)
        axs[0].grid()
        axs[1].grid()
        plt.subplots_adjust(wspace = 0)
        
        # save and close
        if angle==0:
            folder = 'samples_unshifted/'
        else:
            folder = 'samples_shifted/'
        fig.savefig(path_to_figures+folder+col+'_'+str(round(angle,2))+'.png')
        plt.close(fig)
        
       
        





        
