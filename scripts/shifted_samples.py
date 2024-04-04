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


# import Functions
sys.path.append('../../../core_scripts')
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
path_to_save = '../../figures/alhic228_4/iso_animation'
path_to_save_IC = '../../figures/alhic228_4/ic_animation'
path_to_numpy = '../../numpy_files'
path_to_dust = '../../coulter_counter/'
path_to_IC = '../../IC/'


# Import Color Maps
cmap1 = plt.cm.Paired

#%% 1 - import ECM data

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
left = ecmdata(path_to_numpy+'/'+left_file,41)
right = ecmdata(path_to_numpy+'/'+right_file,41)
top = ecmdata(path_to_numpy+'/'+top_file,41)

#%%  2 - import Water Isotope Data

# remove 9th point from Left side - this datapoint is bad
left_wateriso = left_wateriso.drop([8])

#%%  2 - import Dust Data

# Import raw data
stats = pd.read_csv(path_to_dust+'stats_df.csv',header=[0],index_col=[0])
depths = pd.read_csv(path_to_dust+'dust_depths.csv',header=[0],index_col=[0])
volume = pd.read_csv(path_to_dust+'volume_df.csv',header=[0],index_col=[0])

# seperate dataframe into left and right
left_depth = depths[depths.index.str.contains('L')]
right_depth = depths[depths.index.str.contains('R')]
left_stats = stats[stats.index.str.contains('L')]
right_stats = stats[stats.index.str.contains('R')]
left_volume = volume[volume.index.str.contains('L')]
right_volume = volume[volume.index.str.contains('R')]

#%%  2 - import IC Data

# Import raw data
IC = pd.read_csv(path_to_IC+'IC_firstpass.csv',header=[0],index_col=[0])

# seperate dataframe into left and right
left_IC = IC[IC.index.str.contains('L')]
right_IC = IC[(IC).index.str.contains('R')]

#%% Input angles

top_angle = -34.763772174963109
side_angle = (14.432+14.9028)/2




#%% Setup for plot

#  Compute angle
data = top

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

#%% define plot box function

def plotsamps(top,bot,l,r,ax,label,c):

    for i in range(len(top)):
        ax.plot([l,r],[top[i],top[i]],color=c,linewidth=1)
        ax.plot([l,l],[top[i],bot[i]],color=c,linewidth=1)
        ax.plot([r,r],[top[i],bot[i]],color=c,linewidth=1)
        ax.plot([l,r],[bot[i],bot[i]],color=c,linewidth=1)


#%% Water Isotope Plot

if True:
    

        
    
    test_angle = np.linspace(0,top_angle,100)
    #k = 0
    
    for k in range(len(test_angle)):
    
        print("Plotting Angle "+str(test_angle[k]))
        
        
        angle = test_angle[k]
        
        fig,axs = plt.subplots(1,2,dpi=200,figsize=(8,5))
        
        
        # Axis lables and  limits
        axs[0].set_ylabel('Dip-Adjusted Depth (m)')
        axs[0].set_xlabel('Distance Accross Core (mm)')
        axs[0].set_ylim([155.4,155.0])
        axs[0].set_xlim([130,-130])
        
        
        axs[1].set_ylabel('Dip-Adjusted Depth (m)')
        axs[1].set_xlabel('$\delta_{18}O$ (‰)')
        axs[1].set_ylim([155.4,155.0])
        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")
        
        
        
        # Subplot #1 
        for j in range(len(y_vec)):
            
            measy = meas[y_smooth==y_vec[j]]
            depthy = depth[y_smooth==y_vec[j]]
            
            depth_adj = depthy - np.sin(angle*np.pi/180) * (y_vec[j]-middle)/1000
        
            
            for i in range(len(measy)-1):
                
                axs[0].add_patch(Rectangle((y_vec[j]-4.8-middle,depth_adj[i]),9.6,depth_adj[i+1]-depth_adj[i],facecolor=my_cmap(rescale(measy[i]))))
        
        
        # Subplot #1
        # Now working out shifts for water isotopes
        BID_width = 241 #mm
        h20_top_shift = (BID_width-5)/2  * math.tan(side_angle * -1 * (2*math.pi/360)) / 1000
        h20_left_shift = -115  * math.tan(angle  * (2*math.pi/360)) / 1000
        h20_right_shift = 110  * math.tan(angle * (2*math.pi/360)) / 1000
        h20_shift = [h20_left_shift, h20_right_shift, h20_top_shift]
        
        fcnt = 0 
        labels = ["Left","Right","Center"]
        for f in [left_wateriso, right_wateriso, center_wateriso]:
            
            shift = h20_shift[fcnt]
            
            axs[1].plot(f['d18O'],f['mid_depth']+shift,"-o",color=cmap1(fcnt*2+7), label = labels[fcnt]+" - Water Isotope")
        
        
            fcnt+=1
        
        plotsamps(np.array(RISO_top)+h20_shift[1],
                  np.array(RISO_bottom)+h20_shift[1],
                  RISO_left,RISO_right,axs[0],'RISO_',cmap1(1*2+7))
        plotsamps(np.array(LISO_top)+h20_shift[0],
                  np.array(LISO_bottom)+h20_shift[0],
                  LISO_left,LISO_right,axs[0],'LISO_',cmap1(0*2+7))
        plotsamps(CISO_top+h20_shift[2],
                  CISO_bottom+h20_shift[2],
                  CISO_left,CISO_right,axs[0],'CISO_',cmap1(2*2+7))
        
        
        
        # bookkeeping
        axs[0].grid()
        axs[1].grid()
        # bookkeeping
        fig.suptitle('Dip Angle: %.2f degrees'% angle, fontsize=14)
        fig.tight_layout()
        fig.savefig(path_to_save+'/animation_'+str(round(angle,2))+'.png')
        
        plt.close(fig)
        
    #%% Make annimation
    
    
    fps=10
    
    
    
    image_names = []
    for n in test_angle:
        image_names.append('animation_'+str(round(n,2))+'.png')
        
    # allfiles = os.listdir(path_to_save)
    # image_names = [ fname for fname in allfiles if fname.endswith('.png')]
    
    image_files = [os.path.join(path_to_save,img)
                   for img in image_names]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(path_to_save+'/iso_movie.mp4')





#%% Now do IC
############################################################################################

test_angle = [0,top_angle]
#k = 0


ions = ['Cl','NO3','SO4','Na','K','Mg','Ca']
ions_name = ['Cl$^-$','NO$_3^-$','SO$_4^{2+}$','Na$^+$','K$^{+}$','Mg$^{2+}$','Ca$^{2+}$']
for ion,ion_name in zip(ions,ions_name):
    
    print("Running "+ion_name)

    for k in range(len(test_angle)):
    
        print("Plotting Angle "+str(test_angle[k]))
        
        
        angle = test_angle[k]
        
        fig,axs = plt.subplots(1,2,dpi=200,figsize=(8,5))
        
        
        
        
        # Axis lables and  limits
        axs[0].set_ylabel('Dip-Adjusted Depth (m)')
        axs[0].set_xlabel('Distance Accross Core (mm)')
        axs[0].set_ylim([155.4,155.0])
        axs[0].set_xlim([130,-130])
        
        
        axs[1].set_ylabel('Dip-Adjusted Depth (m)')
        axs[1].set_xlabel(ion_name+' Concentration (ppb)')
        axs[1].set_ylim([155.4,155.0])
        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")

        
        
        
        # Subplot 0
        for j in range(len(y_vec)):
            
            measy = meas[y_smooth==y_vec[j]]
            depthy = depth[y_smooth==y_vec[j]]
            
            depth_adj = depthy - np.sin(angle*np.pi/180) * (y_vec[j]-middle)/1000
        
            
            for i in range(len(measy)-1):
                
                axs[0].add_patch(Rectangle((y_vec[j]-4.8-middle,depth_adj[i]),9.6,depth_adj[i+1]-depth_adj[i],facecolor=my_cmap(rescale(measy[i]))))
        
        
        # Subplot #1
        # Now working out shifts for water isotopes
        BID_width = 241 #mm
        dust_left_shift = -95  * math.tan(angle  * (2*math.pi/360)) / 1000
        dust_right_shift = 65  * math.tan(angle * (2*math.pi/360)) / 1000
        dust_shift = [dust_left_shift, dust_right_shift]
        
        
        
        fcnt = 0 
        labels = ["Left","Right"]
        ic_list = [left_IC,right_IC]
        for ic in ic_list:
            
            
            shift = dust_shift[fcnt]
            
            axs[1].plot(ic[ion],ic['Mid Depth']+shift,"-o",color=cmap1(fcnt*2+7))
        
        
            fcnt+=1
        
        # Subplot 9 - boxes
        plotsamps(np.array(RIC_top)+dust_shift[1],
                  np.array(RIC_bottom)+dust_shift[1],
                  RIC_left,RIC_right,axs[0],'RIC_',cmap1(1*2+7))
        plotsamps(np.array(L_top)+dust_shift[0],
                  np.array(L_bottom)+dust_shift[0],
                  L_left,L_right,axs[0],'LIC_',cmap1(0*2+7))
    
        
        
        # bookkeeping
        axs[0].grid()
        axs[1].grid()
        # bookkeeping
        fig.suptitle('Dip Angle: %.2f degrees'% angle, fontsize=14)
        fig.tight_layout()
        fig.savefig(path_to_save_IC+'/'+ion+'_animation_'+str(round(angle,2))+'.png')
        
