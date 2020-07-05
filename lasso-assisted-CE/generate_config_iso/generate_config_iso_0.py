# -*- coding: utf-8 -*-

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
batch_i = 0
# Initialize the cluster object and isomorphs object
iso = lf.isomorphs(mother, dz)
iso.generate_all_iso(config[batch_i], Ec[batch_i], file_index = batch_i)
# For each batch generate isomophoric graphs and save them into json files
#for batch_i in range(len(config)):
    
    