# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:02:07 2019

@author: wangyf
"""
import os
import sys

#os.chdir('..')


import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import pandas as pd
import json
import lattice_functions as lf
import pickle 
import os

font = {'family' : 'normal', 'size'   : 16}
matplotlib.rc('font', **font)

datasize_batch =  [np.load('pi_iso_0.npy').shape[0], 
                   np.load('pi_iso_1.npy').shape[0],
                   np.load('pi_iso_2.npy').shape[0], 
                   np.load('pi_iso_3.npy').shape[0]]

datasize = np.cumsum(datasize_batch)

model_name = 'lasso'
model_batches = ['0', '01', '012', '0123']
model_names = [model_name + '_' + bi for bi in model_batches]
base_dir = os.path.join(os.getcwd(),'cv_results' )
error_per_atom = []
error_per_site = []
n_clusters = []

for mi in model_names:
    [Gcv, J, intercept, RMSE_test_atom, RMSE_test_site] =  pickle.load(open(os.path.join(base_dir, mi,  mi + '.p'), 'rb'))
    n_clusters.append(len(Gcv))
    error_per_atom.append(RMSE_test_atom)
    error_per_site.append(RMSE_test_site *1000) # in meV



#%%
'''
Plot LASSO Accuracy vs number of coefficients selected
'''
fig, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(datasize, error_per_site, 'bo--')
ax1.set_xlabel('Database Size')
ax1.set_xlim([1000, 6500])
ax1.set_ylim([0, 2])
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('RMSE/site (meV)', color='b')
ax1.tick_params('y', colors='b')
ax1.legend(bbox_to_anchor = (1.05, 1),loc= 'upper left', frameon=False)
ax2 = ax1.twinx()
ax2.plot(datasize, n_clusters, 'ro--')
ax2.set_ylabel('# nonzero coefficients', color='r')
ax2.set_ylim([0, 100])
ax2.tick_params('y', colors='r')
#ax2.legend(bbox_to_anchor = (1.3, 1),loc= 'lower left', frameon=False)
fig.tight_layout()
plt.show()
#fig.savefig('elastic_net.png')

#%%
'''
Plot the Pd number distributio sampled in each batch of data
'''
from structure_constants import Ec as Ec_init
from structure_constants import config as config_init

n_all_batch = len(config_init)

NPd_list_batch = []

for batch_i in range(n_all_batch):
    
    # name of the json file
    json_name = 'ES_iso_' + str(batch_i) + '.json'
    # name of the pi file
    pi_name = 'pi_iso_' + str(batch_i) + '.npy'
    with open(json_name) as f:
        ES_data = json.load(f)
        
    Ec_batch_i = ES_data['E_iso']
    config_batch_i = ES_data['config_iso']
    # the number of Pd atoms in each structure
    NPd_list_batch.append(lf.get_NPd_list(config_batch_i))
    
fig = plt.figure(figsize=(6, 4))
n_c, bins, patches = plt.hist(NPd_list_batch[3], 10, facecolor='g', alpha=0.5, width  = 0.9)
plt.ylim([0,500])
plt.xlabel('Number of Pd atoms')  
plt.ylabel('Configuration Counts')  
plt.xticks([0,5,10,15,20])
#fig.savefig('configuration_distribution.png')       
