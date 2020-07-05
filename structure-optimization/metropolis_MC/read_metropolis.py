# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 01:07:46 2019

@author: yifan
"""


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


from ase.visualize import view
from ase.io import read, write
import matplotlib.pyplot as plt


#%%
'''
Simulation conditions
'''
T = 300
rseed = 1
n_total = 10
batch_name = 'metropolis_' + str(T) + 'k_' + str(rseed)


#%%
base_dir = os.getcwd()
input_dir_name = 'metropolis_outputs'
POV_dir_name = 'POV'


input_dir = os.path.join(base_dir, input_dir_name)
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

batch_dir = os.path.join(input_dir, batch_name)
if not os.path.exists(batch_dir):
    os.makedirs(batch_dir)

POV_dir = os.path.join(batch_dir, POV_dir_name)
if not os.path.exists(POV_dir):
    os.makedirs(POV_dir)


def save_POV(batch_name, Pdi, atoms, output_dir):

    pov_args = {
        'transparent': True,  # Makes background transparent. I don't think I've had luck with this option though
        #'run_povray'   : True, # Run povray or just write .pov + .ini files
        'canvas_width': 900,  # Width of canvas in pixels
        #'canvas_height': 500, # Height of canvas in pixels
        'display': False,  # Whether you want to see the image rendering while POV-Ray is running. I've found it annoying
        'rotation': '45x, 0y, -180z',  # Position of camera. If you want different angles, the format is 'ax, by, cz' where a, b, and c are angles in degrees
        'celllinewidth': 0.02,  # Thickness of cell lines
        'show_unit_cell': 0  # Whether to show unit cell. 1 and 2 enable it (don't quite remember the difference)
        # You can also color atoms by using the color argument. It should be specified by an list of length N_atoms of tuples of length 3 (for R, B, G)
        # e.g. To color H atoms white and O atoms red in H2O, it'll be:
        #colors: [(0, 0, 0), (0, 0, 0), (1, 0, 0)]
    }

    # Write to POV-Ray file
    filename = str(batch_name) + '_event_' + str(Pdi) + '_' + '.POV'

    write(os.path.join(output_dir, filename), atoms, **pov_args)


#%%
'''
Load MC results
'''

file_name = batch_name + '.p'
MC_dict = pickle.load(open(os.path.join(input_dir,file_name), "rb"))
config_accepted = MC_dict['accepted']
E_accepted = MC_dict['E']
i_accepted = MC_dict['index']
r_accepted = MC_dict['ratio']

n_accepted = len(i_accepted) - 1
E_pred_final = E_accepted[-1]
config_final = config_accepted[-1]
atoms = energy.append_support(config_final, super_mother, view_flag=True)
print('{} accepted {} final E'.format(n_accepted, E_pred_final))

#%%
i_accepted.append(n_total)
E_accepted.append(E_pred_final)

# get the mean for first x events
section = 1000
r_mean_start = np.mean(r_accepted[:section])

# get the mean for last x events
section = 1000
r_mean_end = np.mean(r_accepted[section:])



#%%
# for index, i in enumerate(i_accepted):

plt.figure(figsize=(6, 6))

fig, ax = plt.subplots()
ax.scatter(i_accepted[: ], E_accepted[: ], s = 2, alpha=1)
plt.xlabel('# moves')
plt.ylabel('Energy $Pd_{20}$ (eV)')
plt.ylim([-25, -5])
plt.xlim([-1000, 11000])

plt.tight_layout()
#fig.savefig(os.path.join(batch_dir, batch_name + '_E_moves_' + str(index) + '.png'))

#%%
'''
'''
# Generate pictures for all accepted configurations

save_POV(batch_name, 0, atoms[0], POV_dir)
