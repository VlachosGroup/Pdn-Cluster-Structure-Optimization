# -*- coding: utf-8 -*-
"""
Set up 5 by 5 FCC abc lattice for Pd cluster expansion
"""

import numpy as np
import lattice_functions as lf



if __name__ == "__main__":
    view_flag = True
else: view_flag = False
    
pos, cell, sitetype = lf.build_pbc_fcc_abc((5, 5), view_flag = False, view_labels = False)

# Calculate the dz
z_pos = pos[:,2]
z_values = np.unique(z_pos)
dz =  np.mean(z_values[1:] - z_values[:-1]) #6**0.5/3 #
mother = pos.copy()

'''
Calculate the nearest neighbor distances based on a center point
'''
origin = mother[82]
NN_distances = np.unique([lf.two_points_D(origin, mi) for mi in mother])


