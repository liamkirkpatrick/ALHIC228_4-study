#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:27:18 2024

@author: Liam
"""

#%% Import packages

import numpy as np
import pandas as pd

#%% User Inputs

path_to_data = '../data/'

#%%  Import Data

# Import raw data
iso = pd.read_excel(path_to_data+'water_iso/228_4_wateriso.xlsx',header=[0],index_col=[0])

# remove 9th point from Left side - this datapoint is bad
iso = iso[iso!='LISO9']

# Import dust data
stats = pd.read_csv(path_to_data+'coulter/stats_df.csv',header=[0],index_col=[0])
depths = pd.read_csv(path_to_data+'coulter/dust_depths.csv',header=[0],index_col=[0])
volume = pd.read_csv(path_to_data+'coulter/volume_df.csv',header=[0],index_col=[0])


# Import IC data
IC = pd.read_csv(path_to_data+'IC/IC_firstpass.csv',header=[0],index_col=[0])

# Import ICMPS data
ICPMS = pd.read_csv(path_to_data+'ICPMS/ICPMS.csv',header=[0],index_col=[0])

# Import CO2 Data
CO2 = pd.read_csv(path_to_data+'gas/CO2.csv',header=[0],index_col=[0])
CH4 = pd.read_csv(path_to_data+'gas/methane.csv',header=[0],index_col=[0])

# Import depth data - raw and dip adjusted
raw_depth = pd.read_csv(path_to_data+'discrete_sampledepths.csv',header=[0],index_col=[0])
dip_depth = pd.read_csv(path_to_data+'dip_adjusted_depths.csv',header=[0],index_col=[0])

#%% Remove depth columns
dfs = [iso,IC,ICPMS,stats,CO2,CH4]
for i in range(len(dfs)):
    dfs[i] = dfs[i].drop(columns = [col for col in dfs[i].columns if 'depth' in col])
    dfs[i] = dfs[i].drop(columns = [col for col in dfs[i].columns if 'Depth' in col])

#%% Make master

samples = pd.concat(dfs, axis=1)

master = pd.concat([raw_depth,dip_depth,samples],axis=1)

master.to_csv(path_to_data+'master_samples.csv')



