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
path_to_isoannimation = '../../figures/iso_annimation/'

# Import Color Maps
cmap1 = plt.cm.Paired

#%% import Data

# toggleon/off
iso_mov = False
plotall = True

#Import ECM
data = ecmdata(path_to_data+'ECM/cmc1-228-4_AC_2023-06-15-15-10.npy',10)



# Import angles
angles = pd.read_csv(path_to_data+'ECM/dip_angle.csv')
top_angle = -angles['Top Angle'].to_numpy().item()
side_angle = -angles['Side Angle'].to_numpy().item()
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
norm = matplotlib.colors.Normalize(vmin=pltmin, vmax=pltmax)

# find y middle
middle = (max(y_vec)-min(y_vec))/2

# samps_to_plot
samps_to_plot = ['CH4']

# clean up labels
label_dict = {'CH4':'CH4 (ppm)'}

# add coordinates
inside_topdepth = np.array([1.5,3.0,5.1,8.5,10.8,13.3,16.4,20.5,23.0])/100+155.065
inside_botdepth = np.array([3.0,5.1,8.5,10.8,13.3,16.4,20.5,23.0,25.9])/100+155.065
outside_topdepth = np.array([0.5,4.9,8.2,10.7,13.2,15.6,18.0,20.5,23.0])/100+155.065+0.015
outside_botdepth = np.array([4.9,8.2,10.7,13.2,15.6,18.0,20.5,23.0,24.7])/100+155.065+0.015


ave_intopage = 15
inside_fromcenter = 0
outside_fromcenter = -50


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

if plotall:

    angle = 0
    print('Plotting ECM Background - angle = '+str(round(angle,2)))
    # Make plot
    ECMfig0,ECMaxs0 = plt.subplots(2,2,dpi=200,height_ratios=[0.8,0.1],figsize=(8,5))
    # Axis 0 lables and  limits
    ECMaxs0[0,0].set_ylabel('Dip-Adjusted Depth (m)')
    ECMaxs0[0,0].set_xlabel('Distance Accross Core (mm)')
    ECMaxs0[0,0].set_ylim([155.4,155.0])
    ECMaxs0[0,0].set_xlim([130,-130])
    # Axis 1 lables and  limits
    ECMaxs0[0,1].set_ylabel('Dip-Adjusted Depth (m)')
    ECMaxs0[0,1].set_ylim([155.4,155.0])
    ECMaxs0[0,1].yaxis.tick_right()
    ECMaxs0[0,1].yaxis.set_label_position("right")
    
    # Subplot #1 
    for j in range(len(y_vec)):
    
        measy = meas[y_smooth==y_vec[j]]
        depthy = depth[y_smooth==y_vec[j]]
    
        depth_adj = depthy - np.tan(-angle*np.pi/180) * (y_vec[j]-middle)/1000
    
    
        for i in range(len(measy)-1):
        
            ECMaxs0[0,0].add_patch(Rectangle((y_vec[j]-4.8-middle,depth_adj[i]),9.6,depth_adj[i+1]-depth_adj[i],facecolor=my_cmap(rescale(measy[i]))))
    
    
    angle = top_angle
    print('Plotting ECM Background - angle ='+str(round(angle,2)))
    # Make plot
    ECMfig1,ECMaxs1 = plt.subplots(2,2,dpi=200,figsize=(8,5),height_ratios=[0.9,0.1])
    # Axis 0 lables and  limits
    ECMaxs1[0,0].set_ylabel('Dip-Adjusted Depth (m)')
    ECMaxs1[0,0].set_xlabel('Distance Accross Core (mm)')
    ECMaxs1[0,0].set_ylim([155.4,155.0])
    ECMaxs1[0,0].set_xlim([130,-130])
    # Axis 1 lables and  limits
    ECMaxs1[0,1].set_ylabel('Dip-Adjusted Depth (m)')
    ECMaxs1[0,1].set_ylim([155.4,155.0])
    ECMaxs1[0,1].yaxis.tick_right()
    ECMaxs1[0,1].yaxis.set_label_position("right")
    
    # Subplot #1 
    for j in range(len(y_vec)):
    
        measy = meas[y_smooth==y_vec[j]]
        depthy = depth[y_smooth==y_vec[j]]
    
        depth_adj = depthy - np.tan(-angle*np.pi/180) * (y_vec[j]-middle)/1000
    
    
        for i in range(len(measy)-1):
        
            ECMaxs1[0,0].add_patch(Rectangle((y_vec[j]-4.8-middle,depth_adj[i]),9.6,depth_adj[i+1]-depth_adj[i],facecolor=my_cmap(rescale(measy[i]))))



#%% loop though all elements

if plotall:

    for col in samps_to_plot:
        
        print('Plotting '+col)
        
        # select appropriate data
        selected_columns = ['rawdepth_top_m','rawdepth_bottom_m',
                            'left_fromcenter_mm','right_fromcenter_mm',
                            'top_fromcenter_mm','bot_fromcenter_mm',col]
    
    
    
        
        for (angle,ECMfig) in zip([0, top_angle],[ECMfig0,ECMfig1]):
            

            
            print("     Angle "+str(round(angle,2)))
            
            
            inside_topdepth_adj = inside_topdepth + ave_intopage/1000*np.tan(side_angle*np.pi/180)+inside_fromcenter/1000*np.tan(angle*np.pi/180) 
            inside_botdepth_adj = inside_botdepth + ave_intopage/1000*np.tan(side_angle*np.pi/180)+inside_fromcenter/1000*np.tan(angle*np.pi/180) 
            outside_topdepth_adj = outside_topdepth + ave_intopage/1000*np.tan(side_angle*np.pi/180)+outside_fromcenter/1000*np.tan(angle*np.pi/180) 
            outside_botdepth_adj = outside_botdepth + ave_intopage/1000*np.tan(side_angle*np.pi/180)+outside_fromcenter/1000*np.tan(angle*np.pi/180) 

            
            # Make plot
            fig = copy.deepcopy(ECMfig)
            axs = fig.axes  # Get the copy of the axes
            axs[0].set_ylabel('Dip-Adjusted Depth (m)')
            fig.suptitle('Depths Adjusted by '+str(round(angle,2))+' degree Dip')
            
            # left pannel
            prefix_index = 0
            color = cmap1(prefix_index*2+7)
            for i in range(len(inside_topdepth)):
                
                l = inside_fromcenter
                r = outside_fromcenter
                TL = inside_topdepth_adj[i]
                TR = outside_topdepth_adj[i]
                BL = inside_botdepth_adj[i]
                BR = outside_botdepth_adj[i]
                
                axs[0].plot([l,r],[TL,TR],color=color,linewidth=1)
                axs[0].plot([l,l],[TL,BL],color=color,linewidth=1)
                axs[0].plot([r,r],[TR,BR],color=color,linewidth=1)
                axs[0].plot([l,r],[BL,BR],color=color,linewidth=1)
                
                
                
            

                
            # # right pannel
            # #prefix_index = 0
            # for index, row in df.iterrows():
                
            #     prefix_index = next((i for i, prefix in enumerate(sticks) if index.startswith(prefix)), None)
                
                
            #     color = cmap1(prefix_index*2+7)
                
            #     axs[1].plot([row[col],row[col]],
            #                 [row['topdepth_dipadj'],row['botdepth_dipadj']],
            #                 color=color)
            # cnt = 0
            # for s in sticks:
            #     index = list(df.index)
            #     df_subset = df[df.index.to_series().str.contains(s)]
            #     #df_subset = df[df.index.to_series() == s]
            #     axs[1].plot(df_subset[col],(df_subset['topdepth_dipadj']+df_subset['botdepth_dipadj'])/2,'-',color=cmap1(cnt*2+7))
            #     cnt+=1
             
            # plot housekeeping
            axs[1].set_xlabel(label_dict.get(col))
            axs[0].grid()
            axs[1].grid()
            axs[2].axis('off')
            axs[3].axis('off')
            plt.subplots_adjust(wspace=0)
            
            #fig.tight_layout()
            
            # add colobar
            left, bottom, width, height = axs[0].get_position().bounds
            cbar_height = 0.02  # Height of the colorbar
            cbar_bottom = bottom - cbar_height - 0.10
            cbar_ax = fig.add_axes([left, cbar_bottom, width, cbar_height])  # adjust these values to fit your layout
            cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=my_cmap), 
                         cax=cbar_ax, orientation='horizontal')
            cb.set_label(label='AC ECM Current (amps)',style='italic',size=8)
    
            plt.subplots_adjust(wspace=0)
    
            
            #fig.tight_layout()
            
            # save and close
            if angle==0:
                folder = 'samples_unshifted/'
            else:
                folder = 'samples_shifted/'
            fig.savefig(path_to_figures+folder+col+'_'+str(round(angle,2))+'_slopedcuts.png')
            plt.close(fig)
        





        
