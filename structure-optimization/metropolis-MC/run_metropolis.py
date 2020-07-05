# -*- coding: utf-8 -*-
"""
Created on Fri May  8 22:40:21 2020

@author: Yifan Wang
"""


'''
Test on Metropolis Monte Carlo
'''


import os
import sys
import pandas as pd
import numpy as np
import pickle
import time

import platform
HomePath = os.path.expanduser('~')
ProjectPath = os.path.join(HomePath, 'Documents', 'GitHub', 
                           'Pdn-Cluster-Structure-Optimization')
# Energy model directory
energy_path = os.path.join(ProjectPath, 'lasso-assisted-CE')

# LASSO model directory
selected_batches = [0, 1, 2, 3]
lasso_model_name = 'lasso' + '_' + ''.join(str(i) for i in selected_batches)
lasso_path = os.path.join(energy_path, lasso_model_name)
lasso_file = os.path.join(lasso_path, lasso_model_name + '.p')


sys.path.append(energy_path)

import energy_functions as energy
from set_ce_lattice import dz
from generate_clusters_super_cell import super_mother

'''
Load energy object
'''
Pdn = energy.Pdn(lasso_file, mother=super_mother, super_cell_flag=True)


#%%
'''
Simulation setup
'''

# start a timer
start = time.time()

# Seed the random
rseed = 1
np.random.seed(rseed)

# number of iterations 
iterations = 10

# Temperature of the simulation in K
T = 300

# Boltzmann constant
kb = 8.617333262145e-05

# Distribution factor from Boltzmann distributin
w = 0



'''
Create initial population 
'''
# Single atoms to put on the base layer
n_seeds = 20


# the index for base layer atoms in super cell
base_indices = np.where(super_mother[:,2] == dz)[0]
base_occ_indices = np.unique(np.random.choice(base_indices, n_seeds, replace = False))


# Initialize the individual configuration in one hot encoding
individual = np.zeros(len(super_mother),dtype = int)
individual[base_occ_indices] = 1

#%%
'''
Predict energy for initial configuration
'''
config_init = energy.one_hot_to_index(individual)
E_pred_init, _  = Pdn.predict_E(config_init)

#%%
'''
MC iterations
'''


x = individual.copy()
E_pred = E_pred_init

accepted = [config_init]
accepted_index = [0] 
accepted_E = [E_pred_init]
acceptance_ratio = [1]
  
for i in range(iterations):
    
    x_new, occ_node_new, _ =  Pdn.swap_occ_empty_fast(x)    
    config_new = energy.one_hot_to_index(x_new)
    E_pred_new, _  = Pdn.predict_E(config_new)
    
    #still need to check minimum distance just in case the new node is closer NN1 to other nodes
    distance_flag = energy.check_Pd_Pd_distance(config_new, super_mother)
    
    # check if the new node is NN1 to existing nodes
    # not sure why this is still needed but when distace = True, this can be false
    NN_flag = energy.check_Pd_Pd_neighboring(occ_node_new, config_new, super_mother)
    
    mass_balance_flag = (np.sum(x_new)==np.sum(x))

    energy_flag = False
    delta_E = E_pred_new - E_pred
    
    # record acceptance ratio
    acceptance_ratio_i = np.min([1, np.exp(-delta_E/kb/T)])
    acceptance_ratio.append(acceptance_ratio_i)
    
    # accept the change if energy going downhill
    if delta_E <= 0: 
        energy_flag = True 
     # test using Boltzmann distribution
    else:
        if T > 0: w = np.exp(-delta_E/kb/T)
        if np.random.rand() <= w: 
            energy_flag = True
                
    if (energy_flag and distance_flag and NN_flag and mass_balance_flag): 
           
        x = x_new
        E_pred = E_pred_new
        accepted.append(config_new)
        accepted_index.append(i+1)
        accepted_E.append(E_pred)

    else:
        pass

    acceptance_count = len(accepted)
    progress = np.around(i/iterations*100, decimals = 3)

    print('{}% done!, {} events, {} accpted, P = {:6.4f}, E = {:6.4f}'.format(progress, i+1,  acceptance_count, acceptance_ratio_i,  E_pred))

  

#%%
#Pdn_atoms,_ = energy.append_support(accepted[-1], super_mother, view_flag = True)

end = time.time()
print("The simulation takes {0:.1f} min".format((end-start)/60))
#%%
'''
Save MC results to a pickle file
'''
base_dir = os.getcwd()
output_dir_name = 'metropolis_outputs'

output_dir = os.path.join(base_dir, output_dir_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
MC_dict = {'accepted': accepted, 'index': accepted_index, 'ratio': acceptance_ratio, 'E': accepted_E }

pickle.dump(MC_dict, open(os.path.join(output_dir_name, 'metropolis_'+str(T)+'k_' + str(rseed) + '.p'),'wb'))



