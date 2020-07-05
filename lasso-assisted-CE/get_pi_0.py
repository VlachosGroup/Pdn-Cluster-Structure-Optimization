# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 14:36:10 2018

@author: wangyf
"""
from set_ce_lattice import mother, dz
#from set_config_constants import config, Ec
import lattice_functions as lf
import numpy as np
import json

batch_i = 0
# name of the json file
json_name = 'ES_iso_' + str(batch_i) + '.json'
# name of the pi file
pi_name = 'pi_iso_' + str(batch_i)
    
  
#%% 
'''
Read the json files
Creat pi matrix
size of number of configuration * numbers of clusters
'''   

empty = 'grey'
filled = 'r'
occ = [empty, filled]

# Initialize the cluster object and isomorphs object
Graphs = lf.initialize_graph_object(mother, dz)

with open('clusters.json') as f:
    Gcv = json.load(f)['Gcv']  
    

with open(json_name) as f:
    ES_data = json.load(f)
    
config_batch_i = ES_data['config_iso']
    
Graphs.get_configs(config_batch_i)
Gsv = Graphs.Gsv


Cal = lf.calculations(occ)
# pi =  Cal.get_pi_matrix_l(Gsv ,Gcv)      

#np.save(pi_name, pi, allow_pickle = True)

pi = np.load('pi_iso_0.npy')

