# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:30:53 2019

@author: wangyf
"""


'''
Save json files of isomophric configurations
'''

from set_ce_lattice import mother, dz
from set_config_constants import config, Ec
import lattice_functions as lf


#%%
'''
Create Configurations in graphs
'''
batch_i = 1
# Initialize the cluster object and isomorphs object
Clusters = lf.initialize_Clusters_object(mother, dz)
iso = lf.isomorphs(mother, dz)
iso.generate_all_iso(config[batch_i], Ec[batch_i], file_index = batch_i)
# For each batch generate isomophoric graphs and save them into json files
#for batch_i in range(len(config)):
    
    

#%%
'''
Create Clusters in graphs
'''
# Initialize subgraph object
#sub = lf.subgraphs(mother, dz)

# Generate cluster upto 3 body interactions by default 
# and save clusters into a json file
#sub.generate_clusters() 
#Gcv1 = sub.get_s2(1)  # 1-body interaction
#Gcv2 = sub.get_s2(2)  # 1-body interaction