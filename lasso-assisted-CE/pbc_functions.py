#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:43:53 2019

@author: wangyifan
"""

'''
test pbc
'''

import lattice_functions as lf
from set_ce_lattice import mother, dz, cell
import numpy as np


#%%
def super_unit_cell(unit_pos, unit_cell, unit_sitetype, extend_rate):
    
    '''
    Extend a unit cell to extend_rate*extend_rate super cell
    '''

    super_cell = unit_cell.copy()
    super_cell[0:2, :] = unit_cell[0:2, :] * extend_rate

    super_pos = unit_pos.copy()
    super_sitetype = unit_sitetype.copy()

    for ri in range(0, extend_rate):
            for rj in range(0, extend_rate):
                if not (ri == 0 and rj == 0):
                    additional_pos = unit_pos.copy() + ri * unit_cell[0]+ rj * unit_cell[1]
                    super_pos = np.concatenate((super_pos, additional_pos))

                    # expand the sitetype too
                    additional_sitetype = unit_sitetype.copy()
                    super_sitetype += additional_sitetype
                    
    return super_pos, super_cell, super_sitetype



#%%
def get_ci_zone(config_i, pos, extend_rate):
    '''
    get which zone is the configuration indices are in
    '''
    
    ci = np.array(config_i)
    ci_zone = []
    
    for i in range(0, extend_rate):
        
        ci_zone_i = []

        
        for j in range(0, extend_rate):
            
            ci_zone_j = []
            
            pos_range_min = i * unit_cell[0] + j* unit_cell[1]
            pos_range_max = (i+1) * unit_cell[0] + (j+1)* unit_cell[1]
            
            pos_range_x = [pos_range_min[0], pos_range_max[0]]
            pos_range_y = [pos_range_min[1], pos_range_max[1]]
            
            for q in ci:
                if pos_range_x[0] <= pos[q][0] <= pos_range_x[1] and pos_range_y[0] <= pos[q][1] <= pos_range_y[1]:
                    ci_zone_j.append(q)
                
            ci_zone_i.append(np.array(ci_zone_j)) 
            
        ci_zone.append(ci_zone_i)
        
    return ci_zone
    


#%%

def extend_unit_cell_with_config(config_i, pos, cell, unit_cell = cell, unit_mother = mother):
    
    '''
    extend unit cell in north, east. northeast and southeast direction to account for periodic boundary conditions
    '''
    
    extend_rate = int(cell[0,0]/unit_cell[0,0])
    
    ci = np.array(config_i)
    ci_zone = get_ci_zone(ci, pos, extend_rate)
    
    extend_pos = pos.copy()
    extend_ci = ci.copy()
    
    for ri in range(0, extend_rate):
        rj = 0
        
        if len(ci_zone[ri][rj]) > 0:
            north_ci  = ci_zone[ri][rj] + len(extend_pos)
            extend_ci = np.concatenate((extend_ci, north_ci))
            
        # Create the north cell, add positions along the y axis and x axis
        north_cell_pos = unit_mother.copy() + cell[1] + ri * unit_cell[0]
        extend_pos = np.concatenate((extend_pos, north_cell_pos))
 
    for rj in range(0, extend_rate):
        ri = 0
        
        if len(ci_zone[ri][rj]) > 0:
            east_ci  = ci_zone[ri][rj] + len(extend_pos)
            extend_ci = np.concatenate((extend_ci, east_ci))
            
        # Create the north cell, add positions along the y axis and x axis
        east_cell_pos = unit_mother.copy() + cell[0] + rj * unit_cell[1]
        extend_pos = np.concatenate((extend_pos, east_cell_pos))
        
        
        
    if len(ci_zone[ri][rj]) > 0:
        north_east_ci  = ci_zone[ri][rj] + len(extend_pos)
        extend_ci = np.concatenate((extend_ci, north_east_ci))
    # Create the north cell, add positions along the y axis and x axis
    ri,rj = 0,0
    north_east_cell_pos = unit_mother.copy() + cell[1] + cell[0]
    extend_pos = np.concatenate((extend_pos, north_east_cell_pos))
    

    if len(ci_zone[ri][rj]) > 0:
        south_east_ci  = ci_zone[ri][rj] + len(extend_pos)
        extend_ci = np.concatenate((extend_ci, south_east_ci))   

    # Create the north cell, add positions along the y axis and x axis
    ri,rj = 0, (extend_rate-1)
    south_east_cell_pos = unit_mother.copy() + cell[0] - unit_cell[1]
    extend_pos = np.concatenate((extend_pos, south_east_cell_pos))
    
    return extend_pos, extend_ci

#%%
    
def extend_unit_cell(pos, cell, unit_cell = cell, unit_mother = mother):
    
    '''
    extend unit cell in north, east. northeast and southeast direction to account for periodic boundary conditions
    '''
    
    extend_rate = int(cell[0,0]/unit_cell[0,0])
    
    extend_pos = pos.copy()
    
    for ri in range(0, extend_rate):
        rj = 0
        
     # Create the north cell, add positions along the y axis and x axis
        north_cell_pos = unit_mother.copy() + cell[1] + ri * unit_cell[0]
        extend_pos = np.concatenate((extend_pos, north_cell_pos))
 
    for rj in range(0, extend_rate):
        ri = 0
            
        # Create the north cell, add positions along the y axis and x axis
        east_cell_pos = unit_mother.copy() + cell[0] + rj * unit_cell[1]
        extend_pos = np.concatenate((extend_pos, east_cell_pos))
        
   # Create the north cell, add positions along the y axis and x axis
    ri,rj = 0,0
    north_east_cell_pos = unit_mother.copy() + cell[1] + cell[0]
    extend_pos = np.concatenate((extend_pos, north_east_cell_pos))
    

    # Create the north cell, add positions along the y axis and x axis
    ri,rj = 0, (extend_rate-1)
    south_east_cell_pos = unit_mother.copy() + cell[0] - unit_cell[1]
    extend_pos = np.concatenate((extend_pos, south_east_cell_pos))
    
    return extend_pos


#%%
'''
Test pbc function
'''
if __name__ == "__main__":
    
    config = [[200]]
    mother_ci = np.array(config[0])
    
    '''
    draw the initial configuration and unit cell
    '''
    
    Graphs = lf.initialize_graph_object(mother, dz)
    Graphs.get_configs(config)
    Gsv = Graphs.Gsv
    lf.drawing(Gsv[0])
    
    '''
    Create a super 2 by 2 lattie
    '''
    unit_pos = mother.copy()
    unit_cell = cell.copy()
    extend_rate = 2
    
    super_pos, super_cell = super_unit_cell(unit_pos, unit_cell, extend_rate)
    ci_zone = get_ci_zone(mother_ci, super_pos, 2)
    
    '''
    Extend the unit cell for pbc with config
    '''
    extend_pos, extend_ci = extend_unit_cell_with_config(mother_ci, unit_pos, unit_cell)       
    lf.plot_CE_lattice(extend_pos)
    extend_config = [list(extend_ci)]
        
    '''    
    Create a graph object and visulize the extend structure
    ''' 
    Graphs = lf.initialize_graph_object(extend_pos, dz)
    Graphs.get_configs(extend_config)   
    Gsv = Graphs.Gsv
    lf.drawing(Gsv[0])
    
#%%
    '''
    Just want to extend the unit cell
    '''
    extend_pos = extend_unit_cell(unit_pos, unit_cell)    
    lf.plot_CE_lattice(extend_pos)


