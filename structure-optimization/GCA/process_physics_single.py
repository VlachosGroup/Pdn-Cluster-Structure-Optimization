# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:43:25 2019

@author: yifan
"""

'''
Read hall of fame output file produced by the genetic algorithm 
Process the individuals and produced the stats of surface physics
Output the information on binding energies and GCNs in csv files
'''

#%% Import session

import os
import sys
import pandas as pd
import numpy as np
from ase.io import read, write
import time

import platform
HomePath = os.path.expanduser('~')
ProjectPath = os.path.join(HomePath, 'Documents', 'GitHub', 
                           'Pdn-Cluster-Structure-Optimization')

# Energy model directory
energy_path = os.path.join(ProjectPath, 'lasso-assisted-CE')
sys.path.append(energy_path)


# LASSO model directory
selected_batches = [0, 1, 2, 3]
lasso_model_name = 'lasso' + '_' + ''.join(str(i) for i in selected_batches)
lasso_path = os.path.join(energy_path, lasso_model_name)
lasso_file = os.path.join(lasso_path, lasso_model_name + '.p')


import energy_functions as energy
import binding_functions as binding

from generate_clusters_super_cell import super_mother
from set_ce_lattice import dz

import GA_functions as GA_f


import matplotlib.pyplot as plt 
import matplotlib
font = {'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2
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


#%% Simulation specifications
T = 300  # Ensemble temperature
nseeds = 55 # Ensemble size
GA = GA_f.generator(lasso_file, mother=super_mother, super_cell_flag=True, T=T, nseeds=nseeds)

# Assign the metrics based on indices 
ncutoff_0 = GA.ncutoff_0
ncutoff_1 = GA.ncutoff_1 # the cutoff between small and large clusters
ncutoff_2 = GA.ncutoff_2

if nseeds < ncutoff_0: 
    floating_flag_index, n_isolate_index, layer_index, E_index, CN_index = 0,1,2,3,4
if (nseeds >= ncutoff_0) and (nseeds < ncutoff_1): 
    floating_flag_index, n_isolate_index, E_index, layer_index, CN_index = 0,1,2,3,4
if nseeds >= ncutoff_1 and nseeds < ncutoff_2: 
    floating_flag_index, E_index, n_isolate_index, CN_index, layer_index = 0,1,2,3,4
if nseeds >= ncutoff_2: 
    floating_flag_index, E_index, n_isolate_index, CN_index, layer_index = 0,1,2,3,4
if nseeds == 55: 
    floating_flag_index, E_index,  CN_index, n_isolate_index, layer_index = 0,1,2,3,4
# The number of Pd atoms in the ensemble
version_list = ['pd20_s0', 'pd20_s1', 'pd20_s2', 'pd20_800k', 'pd20_1300k']
version= 'pd'+str(nseeds) #version_list[4] #
filename_hof = os.path.join(os.getcwd(), version, 'ga_hall_of_fame_' + str(nseeds) + '_' + str(T) + 'k' + '.csv')

# Read the hall of fame file
hof_inds, hof_fitnesses = read_history(filename_hof)


#%% Process the individual to get surface physics 
start_time = time.time()

hof_qualified = []
hof_fitnesses_qualified = []
hof_not_qualified = []
hof_fitnesses_not_qualified = []


for i, (ind_i, fitness) in enumerate(zip(hof_inds, hof_fitnesses)):
    
    # a qualified structure has no floating atoms or isolated atoms
    no_floating_flag = np.abs(fitness[floating_flag_index] - (-1))== 0.0
    no_iso_flag = np.abs(fitness[n_isolate_index]) == 0.0
    
    if no_floating_flag and no_iso_flag:
        hof_qualified.append(ind_i)
        hof_fitnesses_qualified.append(fitness)
    else:
        hof_not_qualified.append(ind_i)
        hof_fitnesses_not_qualified.append(fitness)
        
print('\n{0:6.2f} % individuals qualified\n'.format(len(hof_qualified)/len(hof_inds) *100 ))



#%%
# Only select top 5 as equlibirated structures
no_top_1 = len(hof_qualified)
hof_qualified_top_1 = hof_qualified[:no_top_1]
# Visualize the floating atom objects
# atoms_floating, _ = energy.append_support(GA_f.one_hot_to_index(hof_not_qualified[-1]), super_mother, view_flag=True)

binding_Es = [] # the list for all binding Es of the structures in the ensemble
GCNs = [] # the list for all GCNs of the structures in the ensemble
CN1s = [] # the list for all CN1s of the structures in the ensemble
CN2s = [] # the list for all CN2s  of the structures in the ensemble
ratio_surface = []
GCN_mean = [] # the mean value for GCN of each structure
CN1_mean = [] # the mean value for CN1 of each structure
CN2_mean = [] # the mean value for CN2 of each structure

for i, ind_i in enumerate(hof_qualified_top_1):
    
    
    atoms_i, _ = energy.append_support(GA_f.one_hot_to_index(ind_i), super_mother, view_flag=False)
    binding_Es_i, _, _, _, GCNs_i, CN1s_i, CN2s_i, ratio_surface_i =  binding.predict_binding_Es_fast(atoms_i, output_descriptor= True, view_flag = False, top_only = True)
    binding_Es += list(binding_Es_i)
    GCNs += list(GCNs_i)
    CN1s += list(CN1s_i)
    CN2s += list(CN2s_i)
    ratio_surface.append(ratio_surface_i)
    GCN_mean.append(np.mean(GCNs_i))
    CN1_mean.append(np.mean(CN1s_i))
    CN2_mean.append(np.mean(CN2s_i))
    print('{0:6.2f} % in evaluating surface physics done'.format((i+1)*100/no_top_1))
    

# Take out the zero binding enegry sites
binding_Es =  np.array(binding_Es)
GCNs = np.array(GCNs)
CN1s = np.array(CN1s)
CN2s = np.array(CN2s)

# Pad the variables with 0s, just for pandas saving purpose
ratio_surface_padded = np.array([np.nan] * len(binding_Es))
ratio_surface_padded[:len(ratio_surface)] = np.array(ratio_surface)

GCN_mean_padded =  np.array([np.nan] * len(binding_Es))
GCN_mean_padded[:len(GCN_mean)] = np.array(GCN_mean)

CN1_mean_padded =  np.array([np.nan] * len(binding_Es))
CN1_mean_padded[:len(CN1_mean)] = np.array(CN1_mean)

CN2_mean_padded = np.array([np.nan] * len(binding_Es))
CN2_mean_padded[:len(CN2_mean)] = np.array(CN2_mean)


#%%
filename_physics = os.path.join(os.getcwd(), version, 'Pd'+ str(nseeds) + '_' + str(T) + 'k'  + '_physics.csv')
raw_df = {'binding E': binding_Es,
            'GCN': GCNs,
            'CN1': CN1s,
            'CN2': CN2s,
            'ratio surface': ratio_surface_padded,
            'GCN mean': GCN_mean_padded,
            'CN1 mean': CN1_mean_padded,
            'CN2 mean': CN2_mean_padded
            }

# Convert to a dataframe
df = pd.DataFrame(raw_df)
# Delete the file generated previously
if os.path.exists(filename_physics):  os.remove(filename_physics)
with open(filename_physics, 'a') as f:
    df.to_csv(f, header=f.tell()==0)
    
end_time = time.time()
min_time = (end_time - start_time) / 60
print('\nThe processing takes {0:.4f} minutes'.format(min_time))


