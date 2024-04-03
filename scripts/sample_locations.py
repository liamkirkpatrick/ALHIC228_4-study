#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:25:49 2024

Make a plot showing where the samples are cut from ALHIC2201

@author: Liam
"""

#%% import packages

#utility
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle


#%% User inputs

# filepaths
path_to_figures = '../../figures/sample_locations/'
path_to_data = '../data/'

# plot bounds
pltmin = 1.3*10**(-8)
pltmax = 2.0*10**(-8)

# Dip angles (not adjusting top face, but do have to adjust for into page)
side_angle = -(14.432+14.9028)/2

#%% establish colorscale 

# make colormap
cmap = matplotlib.colormaps.get_cmap('coolwarm')

#%% Import data

data = np.load(path_to_data+'ECM/cmc1-228-4_AC_2023-06-15-15-10.npy')
y_vec = np.unique(data[:,2])



#%% Import depth data

df = pd.read_csv(path_to_data+'discrete_sampledepths.csv')

# adjust depth for into/pout of page
ave_intopage = (df['top_fromcenter_mm']+df['bot_fromcenter_mm'])/2
df['topadj'] = df['rawdepth_top_m'] + ave_intopage/1000 * np.sin(side_angle*np.pi/180)
df['botadj'] = df['rawdepth_bottom_m'] + ave_intopage/1000 * np.sin(side_angle*np.pi/180)


#%% smooth data

# number of points to smooth. 41mm is equal to 1cm
window = 41 # must be odd

#make empty smooth arrays (to be filled later)
x_smooth = np.array([])
y_smooth = np.array([])
meas_smooth = np.array([])
button_smooth = np.array([])
depth_smooth = np.array([])

# loop through all tracks (y_vec falues)
for i in y_vec:

    ind = (data[:,2] == i)
    x = data[ind,0]
    depth = data[ind,1]
    y = data[ind,2]
    meas = data[ind,3]
    button = data[ind,4]
    
    # loop through all of the depths within that vector, leaving space at each end
    # to ensure all smooth values can averagle half of the window on either side
    for j in range(int(((window-1)/2)), int(sum(ind)-(window-1)/2+1)):
        
        
        x_smooth = np.append(x_smooth, x[j])
        depth_smooth = np.append(depth_smooth,depth[j])
        y_smooth = np.append(y_smooth,i)
        
        if sum(button[int(j-(window-1)-2):int(j+(window-1)/2)])>0:
            button_smooth = np.append(button_smooth,1)
        else:
            button_smooth = np.append(button_smooth,0)
            
        meas_smooth = np.append(meas_smooth,np.mean(meas[int(j-((window-1)/2)):int(j+((window-1)/2))]))
        
#%% Define plotting function

def plotsamps(df,ax,label):
    
    # list all boxes
    TL = df['topadj'].to_numpy()
    TR = df['topadj'].to_numpy()
    BL = df['botadj'].to_numpy()
    BR = df['botadj'].to_numpy()
    l = df['left_fromcenter_mm'].to_numpy()
    r = df['right_fromcenter_mm'].to_numpy()
    
    print('TL = '+str(TL))
    print('BL = '+str(BL))
    print('l = '+str(l))
    print('r = '+str(r))

    # plot each box
    for i in range(len(TL)):
        ax.plot([l[i],r[i]],[TL[i],TR[i]],color='k',linewidth=1)
        ax.plot([l[i],l[i]],[TL[i],BL[i]],color='k',linewidth=1)
        ax.plot([r[i],r[i]],[TR[i],BR[i]],color='k',linewidth=1)
        ax.plot([l[i],r[i]],[BL[i],BR[i]],color='k',linewidth=1)
        
        ax.text((l[i]+r[i])/2,(TL[i]+TR[i]+BL[i]+BR[i])/4,label+'_'+str(i+1),ha='center')
        

#%% Make first plot (ISO/IC/ICPMS)

fig2,axs2 = plt.subplots(1,1,figsize = (12,12),dpi=200)

fig2.tight_layout(pad=5.0)

# set up colors
colors = ([0,0,1],[1,0,0])
my_cmap = matplotlib.colormaps['Spectral']
rescale = lambda k: (k-pltmin) /  (pltmax-pltmin)

# Axis lables and  limits
axs2.set_ylabel('Depth (m)')
axs2.set_xlabel('Distance Accross Core (mm)')

#axs2.invert_yaxis()
axs2.invert_xaxis()
axs2.set_ylim([155.38,155.04])
axs2.set_xlim([130,-130])
axs2.set_title('ALHIC1901 228_4 IC / ICMPS / Dust / Water Iso Locations', fontsize=20)

# find y middle
middle = (max(y_vec)-min(y_vec))/2

# plot ECM
for j in range(len(y_vec)):
    
    meas = meas_smooth[y_smooth==y_vec[j]]
    depth = depth_smooth[y_smooth==y_vec[j]]
    
    print('Y = '+str(y_vec[j]))
    
    for i in range(len(meas)-1):
        
        axs2.add_patch(Rectangle((y_vec[j]-4.8-middle,depth[i]),9.6,depth[i+1]-depth[i],facecolor=my_cmap(rescale(meas[i])))) 


#plot all samples
samps = ['LIC','RIC','ICPMS','RISO','LISO','CISO']
for s in samps:
   plotsamps(df[df['ID'].str.contains(s)],axs2,s)
    
    

# save figure
fig2.savefig(path_to_figures+'impurity_sample_locations.png')

#%% Make 2nd plot

fig3,axs3 = plt.subplots(1,1,figsize = (12,12),dpi=200)

fig3.tight_layout(pad=5.0)


# set up colors
colors = ([0,0,1],[1,0,0])
my_cmap = matplotlib.colormaps['Spectral']
rescale = lambda k: (k-pltmin) /  (pltmax-pltmin)

# Axis lables and  limits
axs3.set_ylabel('Depth (m)')
axs3.set_xlabel('Distance Accross Core (mm)')

#axs2.invert_yaxis()
axs3.invert_xaxis()
axs3.set_ylim([155.38,155.04])
axs3.set_xlim([130,-130])
axs3.set_title('ALHIC1901 228_4 Gas Sample Locations', fontsize=20)

# find y middle
middle = (max(y_vec)-min(y_vec))/2

# plot ECM
for j in range(len(y_vec)):
    meas = meas_smooth[y_smooth==y_vec[j]]
    depth = depth_smooth[y_smooth==y_vec[j]]

    
    print('Y = '+str(y_vec[j]))
    
    for i in range(len(meas)-1):
        
        axs3.add_patch(Rectangle((y_vec[j]-4.8-middle,depth[i]),9.6,depth[i+1]-depth[i],facecolor=my_cmap(rescale(meas[i])))) 


# plot all samples
samps = ['LOSU','ROSU','LSIO','RSIO']
for s in samps:
    plotsamps(df[df['ID']==s],axs3,s)


fig3.savefig(path_to_figures+'gas_sample_locations.png')




