# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:02:19 2020

@author: Yifan Wang
"""


'''
Test file for CE model usage 
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

from set_config_constants import config
from set_ce_lattice import mother
import energy_functions as energy
from set_ce_lattice import dz
from generate_clusters_super_cell import super_mother


#%% Predict the energy for structures in DFT dataset on a 5 by 5 lattice (mother)

# Load energy object
Pdn = energy.Pdn(lasso_file, mother=mother, super_cell_flag=False)

# select a Pd single atom from configuration dataset
config_SA = config[0][0]

# Predict energy for a single atom
E_pred_SA, _  = Pdn.predict_E(config_SA)

# Visualize the atomic configuration
atoms_SA = energy.append_support(config_SA, mother, view_flag=True)

#%% Generate a random structure and calculate its energy on a 10 by 10 lattice (super_mother)

# Load energy object
Pdn_super = energy.Pdn(lasso_file, mother=super_mother, super_cell_flag=True)

# Single atoms to put on the base layer
n_seeds = 20

# the index for base layer atoms in super cell
base_indices = np.where(super_mother[:,2] == dz)[0]
base_occ_indices = np.unique(np.random.choice(base_indices, n_seeds, replace = False))


# Initialize the individual configuration in one hot encoding
rnd_individual = np.zeros(len(super_mother),dtype = int)
rnd_individual[base_occ_indices] = 1


# Predict energy for initial configuration
config_rnd = energy.one_hot_to_index(rnd_individual)
E_pred_rnd, _  = Pdn_super.predict_E(config_rnd)

# Visualize the atomic configuration
atoms_rnd = energy.append_support(config_rnd, super_mother, view_flag=True)


