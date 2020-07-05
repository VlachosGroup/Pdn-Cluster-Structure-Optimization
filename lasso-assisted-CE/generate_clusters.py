# -*- coding: utf-8 -*-


'''
Save json files of clusters 
'''

from set_ce_lattice import mother, cell, dz, NN_distances
from set_config_constants import config, Ec
import lattice_functions as lf
import pbc_functions as pbc

'''   
Set cutoff distance
'''
cutoff_at_NN = 10
cutoff_distance = NN_distances[cutoff_at_NN]
#%%
'''
Create Clusters in graphs in  mother unit cell
'''

if __name__ == "__main__":
    
# Initialize subgraph object
    sub = lf.subgraphs(mother, dz)
    
    # Generate cluster upto 3 body interactions by default 
    # and save clusters into a json file
    sub.generate_clusters(cutoff_distance) 



