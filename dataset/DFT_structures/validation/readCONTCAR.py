# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 14:15:41 2018

@author: Yifan Wang
Read all CONTCAR files in the current directory 
Save as POV files
"""
import os 
import sys
from ase.io import read, write
from ase.visualize import view
import glob


files = []
for file in glob.glob("*CONTCAR"):
    files.append(file)


filename = files[0]

atoms = read(filename)
atoms = atoms.repeat((2,1,1))
view(atoms)



#%%
base_dir = os.getcwd()
POV_dir_name = 'POV'
POV_dir = os.path.join(base_dir,  POV_dir_name)


#%%
def save_POV(Pdi, index, atoms, output_dir):

    pov_args = {
    	'transparent': True, #Makes background transparent. I don't think I've had luck with this option though
        'canvas_width': 1800., #Size of the width. Height will automatically be calculated. This value greatly impacts POV-Ray processing times
        'display': False, #Whether you want to see the image rendering while POV-Ray is running. I've found it annoying
        'rotation': '0x, 0y,90z', #Position of camera. If you want different angles, the format is 'ax, by, cz' where a, b, and c are angles in degrees
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
    
for file_i in files:
    Pdi = file_i.split('-')[0].replace('Pd','')
    index = file_i.split('-')[1]
    atoms = read(file_i)
    atoms = atoms.repeat((2,1,1))
    save_POV(Pdi, index, atoms, POV_dir)