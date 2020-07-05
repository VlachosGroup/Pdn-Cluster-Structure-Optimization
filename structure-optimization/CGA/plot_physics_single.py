# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:55:00 2019

@author: yifan
"""

'''
Read hall of fame binding energies and GCNs in csv files
Produce histograms
'''
#%% Import session

import os
import sys
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt 
import matplotlib
font = {'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
dpi = 300.
matplotlib.rcParams['figure.dpi'] = dpi

#%% User define functions

def read_history(filename):
    '''
    Read the history type csv output files
    return the population in a list 
    return the fitnesses in a numpy array
    '''
    # Convert to a dataframe
    df = pd.read_csv(filename, index_col=False)
    
    # Convert the pandas value to one hot encoding
    population = []
    for indi in df['individual']:
        population.append([int(i) for i in indi.strip('][').split(', ')])
    
    # Save the fitness value into a 2D array
    fitnesses = np.column_stack((df['fitness1'], df['fitness2'], df['fitness3'], df['fitness4'], df['fitness5']))
    
    return population, fitnesses

def read_physics(filename):
    '''
    Read physics csv file
    return binding energies and GCNs as arrays
    '''
    # Convert to a dataframe
    df = pd.read_csv(filename, index_col=False)
    binding_Es = np.array(df['binding E'])
    GCNs = np.array(df['GCN'])
    CN1s = np.array(df['CN1'])
    CN2s = np.array(df['CN2'])

    return binding_Es, GCNs, CN1s, CN2s


#%% Simulation specifications
T = 300  # Ensemble temperature
nseeds = 20 # Ensemble size
#GA = GA_f.generator(lasso_file, mother=super_mother, super_cell_flag=True, T=T, nseeds=nseeds)

ncutoff = 15 # the cutoff between small and large clusters
if nseeds <= ncutoff: E_index, CN_index = 3, 4
else: E_index, CN_index = 1, 3
# The number of Pd atoms in the ensemble
version_list = ['pd5', 'pd10', 'pd15', 'pd20_s0', 'pd20_s1', 'pd20_s2', 'pd20_800k', 'pd20_1300k', 'pd25', 'pd30']
version= version_list[4]
filename_hof = os.path.join(os.getcwd(), version, 'ga_hall_of_fame_' + str(nseeds) + '_' + str(T) + 'k' + '.csv')

# Set the ensemble size as the population size
n_qualified = 5


#%% Plot the physics
filename_physics = os.path.join(os.getcwd(),  version, 'Pd'+ str(nseeds) + '_' + str(T) + 'k'  + '_physics.csv')
binding_Es, GCNs, CN1s, CN2s = read_physics(filename_physics)

# Specifics for histograms
range_binding_E = (-4, 0)  # The min binding E is -4 
range_GCN = (0, 11) # The max GCN is 10.5
bin_size = 40
norm_color = plt.Normalize(vmin=0, vmax= 0.05) 

# Normalize the data
norm_factor = 1/nseeds/n_qualified # total number of atoms
weights = norm_factor * np.ones(binding_Es.shape)

def plot_CN(CNs, range_CN, weights,
             T_single = T, nseeds = nseeds, version = version, ):
    '''
    Histogram 1, GCN
    ''' 
    fig, ax = plt.subplots(figsize= (6,6))
    ax.hist(CNs, bins=bin_size, weights = weights, range = range_CN, 
            color = 'steelblue',  alpha = 0.5, 
            edgecolor = 'k', linewidth = 0.5)
    ax.set_xlabel('CN')
    ax.set_ylabel('Normalized Frequency')
    plt.tight_layout()
    fig.savefig(os.path.join(os.getcwd(), version, 'Pd'+ str(nseeds) + '_' + str(T) + 'k' + '_GCN.PNG'))


def plot_binding_E(binding_Es, range_binding_E, weights, 
                   T_single = T, nseeds = nseeds, version = version):
    '''
    Histogram 2, CO binding E
    '''
    fig, ax = plt.subplots(figsize= (6,6))
    ax.hist(binding_Es, bins=bin_size, weights = weights, range = range_binding_E, 
            color = 'steelblue', alpha = 0.5,
            edgecolor = 'k', linewidth = 0.5)
    ax.set_xlabel('CO ' + r'$\rm E_{binding} (eV)$')
    ax.set_ylabel('Normalized Frequency')
    plt.tight_layout()
    fig.savefig(os.path.join(os.getcwd(), version, 'Pd'+ str(nseeds) + '_' + str(T) + 'k' + '_binding.PNG'))


def plot_hist2d(GCNs, binding_Es, range_GCN, range_binding_E, weights, 
                T_single = T, nseeds = nseeds, version = version):
    '''
    Histogram 2D, GCN bs CO binding E
    '''
    fig, ax = plt.subplots(figsize= (6,6))
    h, _, _, _ = ax.hist2d(GCNs, binding_Es, bins = bin_size, weights = weights, 
                           range = [range_GCN, range_binding_E], 
                           cmap='Blues',
                           norm=norm_color)
    ax.set_xlabel('GCN')
    ax.set_ylabel('CO ' + r'$\rm E_{binding} (eV)$')
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm_color)
    cb = plt.colorbar(sm)
    cb.set_label('Normalized Frequency')
    #plt.tight_layout()
    fig.savefig(os.path.join(os.getcwd(), version, 'Pd'+ str(nseeds) + '_' + str(T) + 'k' + '_hist2D.PNG'))
    
#GCN    
plot_CN(GCNs, range_GCN, weights)
#CN1
plot_CN(CN1s, range_GCN, weights)
#CN2
plot_CN(CN2s, range_GCN, weights)
