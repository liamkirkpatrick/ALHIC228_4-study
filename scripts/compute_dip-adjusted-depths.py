#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:51:22 2024

@author: Liam
"""

#%% Import packages
import numpy as np
import pandas as pd


#%% User Inputs

path_to_data = '../data/'

#%% load data

angles = pd.read_csv(path_to_data+'ECM/dip_angle.csv')
top_angle = -angles['Top Angle'].to_numpy()
side_angle = angles['Top Angle'].to_numpy()
df = pd.read_csv(path_to_data+'discrete_sampledepths.csv')

#%% Compute Dip adjusted depths

ave_intopage = (df['top_fromcenter_mm']+df['bot_fromcenter_mm'])/2
ave_crosspage = (df['left_fromcenter_mm']+df['right_fromcenter_mm'])/2
adj = pd.DataFrame()
adj['ID'] = df['ID']
adj['topdepth_dipadj'] = (df['rawdepth_top_m'] 
                   + ave_intopage/1000 * np.sin(side_angle*np.pi/180)
                   + ave_crosspage/1000 * np.sin(top_angle*np.pi/180))
adj['middepth_dipadj'] = ((df['rawdepth_top_m']+df['rawdepth_bottom_m'])/2
                   + ave_intopage/1000 * np.sin(side_angle*np.pi/180)
                   + ave_crosspage/1000 * np.sin(top_angle*np.pi/180))
adj['botdepth_dipadj'] = (df['rawdepth_bottom_m'] 
                   + ave_intopage/1000 * np.sin(side_angle*np.pi/180)
                   + ave_crosspage/1000 * np.sin(top_angle*np.pi/180))

#%% Save adjusted depths

adj.to_csv(path_to_data+'dip_adjusted_depths.csv',index=False)