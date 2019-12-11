# -*- coding: utf-8 -*-
"""
Created on Thu Aug 1 2019

@author: wangyf
This function is only used for plotting purpose

"""
import matplotlib.pyplot as plt
import lattice_functions as lf
import numpy as np
import json
import networkx as nx
import os 
from set_ce_lattice import dz, mother, sitetype
from set_config_constants import config, Ec


empty = 'grey'
filled = 'r'
occ = [empty, filled]

'''
only draw 1st nearest neighbors?
'''
NN1 = 1
'''
Draw mother/conifgurations/clusters?
'''
draw = [1, 0, 0]


Graphs = lf.graphs(occ, NN1, draw)
Graphs.get_mother(mother, dz)
Gm = Graphs.Gm
iso_flag = False
batch_i = 0

if iso_flag:
    with open('ES_iso_' + str(batch_i) +'.json') as f:
        ES_data = json.load(f)
        
    Ec = ES_data['E_iso']
    config = ES_data['config_iso']
    
else: 
    
    config_unique = config[0] + config[1] + config[2] + config[3]
    Ec_unique = Ec[0] + Ec[1] + Ec[2] + Ec[3]
    config = config_unique.copy()
    Ec = Ec_unique.copy()
    
#%%
'''
Create Configurations
'''
Graphs.get_configs(config)
Gsv = Graphs.Gsv


#%%
'''
Plot selected configurations
'''
indices = [0, 4, 11, 45, 95]

base_dir = os.getcwd()
output_dir_name = 'selected_configs'
POV_dir_name = 'POV'
lat_dir_name = 'Lattice'

POV_dir = os.path.join(base_dir, output_dir_name, POV_dir_name)
if not os.path.exists(POV_dir): os.makedirs(POV_dir)    

lat_dir = os.path.join(base_dir, output_dir_name, lat_dir_name)
if not os.path.exists(lat_dir): os.makedirs(lat_dir)    

    
for i, ind in enumerate(indices):
    
   fig, _ = lf.drawing3D(Gsv[ind], mother, sitetype) 
   filename = 'Pd'+str(len(config[ind])) +'-' + str(i) + '.png'
   fig.savefig(os.path.join(lat_dir, filename))
   plt.close()

#%%
'''
Draw the configurations in ASE
'''
from ase.visualize import view
from ase.io import read, write
from ase.data import covalent_radii
from ase import Atom, Atoms
from ase.build import surface
import energy_functions as energy

'''
Save the atom object
'''

def append_support(ind_index, view_flag = False):
    '''
    take in the 1/0 vector for index
    '''
    Pdr = covalent_radii[46]
    Or = covalent_radii[8]
    PdPd = Pdr*2
    PdO = 2.1 # the average PdO length take from CONTCAR files 

    
    def ceria():
        #Lattice constant
        a = 5.49
        CeO2 = Atoms('Ce4O8', scaled_positions =[ (0., 0., 0.),
                      (0., 0.5, 0.5),
                      (0.5, 0., 0.5),
                      (0.5, 0.5, 0.),
                      (0.75, 0.25, 0.25),
                      (0.25, 0.75, 0.75),
                      (0.75, 0.75, 0.75),
                      (0.25, 0.25, 0.25),
                      (0.25, 0.25, 0.75),
                      (0.75, 0.75, 0.25),
                      (0.25, 0.75, 0.25),
                      (0.75, 0.25, 0.75)],
                      cell = [a,a,a],
        			  pbc = True	)
        #Scales the atomic positions with the unit cell
        #CeO2.set_cell(cell, scale_atoms=True)
        #(1,1,1) is the slab type. There are 2 unit cells along 
        #z direction
        slab = surface(CeO2, (1, 1, 1), 2)
        
        #Repeating the slab 2 unit cells in x and 1 unit cell 
        #in y directions
        slab = slab.repeat((2,2,1))
        slab.center(vacuum=10, axis=2)
        del slab[[atom.index for atom in slab if atom.z>15]]
        
        return slab
    
    support = ceria()
    #set origin value by looking at the ase GUI, pick one oxygen atom
    origin_index = 7 
    origin = support[origin_index].position.copy() 
    origin[2] = origin[2] +  PdO + 1.5
    
    # superpose the Pd lattice onto ceria lattice
    mother_with_support = origin + (mother- mother[0]) * PdPd 
    # select the occupied nodes
    Pdpos= mother_with_support[ind_index]
    
    #PdNP = Atoms('Pd36', positions = Pdm)
    
    nPd = len(Pdpos)
    for i in range(nPd):
        support.append(Atom('Pd', position = Pdpos[i]))
    
    if view_flag: view(support) 
    
    return support


def save_CONTCAR(Pdi, index, atoms, output_dir):

    filename = 'Pd'+str(Pdi) +'-' + str(index) + '-CONTCAR'
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)    
    write(os.path.join(output_dir, filename), atoms)


def save_POV(Pdi, index, atoms, output_dir):

    pov_args = {
    	'transparent': True, #Makes background transparent. I don't think I've had luck with this option though
        'canvas_width': 900., #Size of the width. Height will automatically be calculated. This value greatly impacts POV-Ray processing times
        'display': False, #Whether you want to see the image rendering while POV-Ray is running. I've found it annoying
        'rotation': '45x, 0y,180z', #Position of camera. If you want different angles, the format is 'ax, by, cz' where a, b, and c are angles in degrees
        'celllinewidth': 0.02, #Thickness of cell lines
        'show_unit_cell': 0 #Whether to show unit cell. 1 and 2 enable it (don't quite remember the difference)
        #You can also color atoms by using the color argument. It should be specified by an list of length N_atoms of tuples of length 3 (for R, B, G)
        #e.g. To color H atoms white and O atoms red in H2O, it'll be:
        #colors: [(0, 0, 0), (0, 0, 0), (1, 0, 0)]
        }

    #Write to POV-Ray file
    filename = 'Pd'+str(Pdi) +'-' + str(index) + '.POV'
    if not os.path.exists(output_dir): os.makedirs(output_dir)    
    write(os.path.join(output_dir, filename), atoms, **pov_args)
    
config_selected = [config[i] for i in indices]
atoms_list = [append_support(si) for si in config_selected]

index = 0

for i, config_i in enumerate(config_selected):
    
    #save_CONTCAR(len(config_i), index, atoms_list[i], CONTCAR_dir)
    save_POV(len(config_i), index, atoms_list[i], POV_dir)
    index += 1

#view(atoms_list[3])
    
