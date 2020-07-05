# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 22:29:18 2019

@author: yifan
"""

'''
Read the csv file produced by the genetic algorithm
'''

#%% Import session


import os
import sys
import pandas as pd
import numpy as np
from ase.io import read, write
import time

import platform
HomePath = os.path.expanduser('~')
ProjectPath = os.path.join(HomePath, 'Documents', 'GitHub', 
                           'Pdn-Cluster-Structure-Optimization')

# Energy model directory
energy_path = os.path.join(ProjectPath, 'lasso-assisted-CE')
sys.path.append(energy_path)


# LASSO model directory
selected_batches = [0, 1, 2, 3]
lasso_model_name = 'lasso' + '_' + ''.join(str(i) for i in selected_batches)
lasso_path = os.path.join(energy_path, lasso_model_name)
lasso_file = os.path.join(lasso_path, lasso_model_name + '.p')


import energy_functions as energy

from generate_clusters_super_cell import super_mother
from set_ce_lattice import dz

import GA_functions as GA_f


import matplotlib.pyplot as plt 
import matplotlib
font = {'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2
dpi = 300.
matplotlib.rcParams['figure.dpi'] = dpi

#%% User define functions

def read_history(filename):
    '''
    Read the history type csv output files
    return the population in a list 
    return the fitnesses in a numpy array
    '''
    # Convert to a dataframe
    df = pd.read_csv(filename, index_col=False)
    
    # Convert the pandas value to one hot eqncoding
    population = []
    for indi in df['individual']:
        population.append([int(i) for i in indi.strip('][').split(', ')])
    
    # Save the fitness value into a 2D array
    fitnesses = np.column_stack((df['fitness1'], df['fitness2'], df['fitness3'], df['fitness4'], df['fitness5']))
    
    return population, fitnesses

def read_stats(filename):
    '''
    Read the stats type csv output files
    return the stats
    '''
    # Convert to a dataframe
    df = pd.read_csv(filename, index_col=False)  
    generation_no = np.array(df['generation no.'])
    mean = np.array(df['mean'])  
    sd = np.array(df['sd'])
    min_val = np.array(df['min_val'])  
    max_val = np.array(df['max_val'])
    
    
    return generation_no, mean, sd, min_val, max_val
    
    
#%% Simulation specifications
T = 5000  # Ensemble temperature
nseeds = 20 # Ensemble size
GA = GA_f.generator(lasso_file, mother=super_mother, super_cell_flag=True, T=T, nseeds=nseeds)

# Assign the metrics based on indices 
ncutoff_0 = GA.ncutoff_0
ncutoff_1 = GA.ncutoff_1 # the cutoff between small and large clusters
ncutoff_2 = GA.ncutoff_2

if nseeds < ncutoff_0: 
    floating_flag_index, n_isolate_index, layer_index, E_index, CN_index = 0,1,2,3,4
if (nseeds >= ncutoff_0) and (nseeds < ncutoff_1): 
    floating_flag_index, n_isolate_index, E_index, layer_index, CN_index = 0,1,2,3,4
if nseeds >= ncutoff_1 and nseeds < ncutoff_2: 
    floating_flag_index, E_index, n_isolate_index, CN_index, layer_index = 0,1,2,3,4
if nseeds >= ncutoff_2: 
    floating_flag_index, E_index, n_isolate_index, CN_index, layer_index = 0,1,2,3,4
if nseeds == 55: 
    floating_flag_index, E_index,  CN_index, n_isolate_index, layer_index = 0,1,2,3,4

# The number of Pd atoms in the ensemble
version_list = ['pd20_s0', 'pd20_s1', 'pd20_s2', 'pd20_800k', 'pd20_1300k', 'pd20_5000k']
version= version_list[-1] #'pd' + str(nseeds) 

filename_hof = os.path.join(os.getcwd(), version, 'ga_hall_of_fame_' + str(nseeds) + '_' + str(T) + 'k' + '.csv')
filename_best = os.path.join(os.getcwd(), version, 'ga_generation_best_' + str(nseeds) + '_' + str(T) + 'k' + '.csv')
filename_stats = os.path.join(os.getcwd(), version, 'ga_stats_output_' + str(nseeds) + '_' + str(T) + 'k' + '.csv')
#%% Plot and save the results

def plot_stats(filename = filename_stats, nseeds = nseeds, version = version, max_gen = None):
    
    '''
    Read stats
    '''
    generation_no, mean, sd, min_val, max_val = read_stats(filename_stats)
    if max_gen == None:
        max_gen = max(generation_no)
    fig, ax = plt.subplots(figsize= (6,6))
    ax.plot(generation_no, min_val, 'steelblue')
    ax.plot(generation_no, max_val, 'steelblue', linewidth = 0.8)
    ax.fill_between(generation_no, min_val, max_val, color = 'steelblue', alpha = 0.3)
    ax.set_xlabel('# Generation')
    ax.set_ylabel('Energy (eV)')
    ax.set_xlim(0, max_gen)
    if max_gen > 1000:
        # Set the log scale
        ax.set_xscale('log')
        ax.set_xlim(1, max_gen)
        ax.set_xticks([1,10,100,1000,max_gen])
    plt.savefig(os.path.join(os.getcwd(), version, 'Pd'+ str(nseeds) + '_' + str(T) + 'k' +  '_gen_' + str(max_gen) + '_stats.PNG'))

def plot_hof(filename = filename_hof, nseeds = nseeds, version = version):
    
    '''
    Read hall of fame
    '''
    # Select the best one
    hof, hof_fitnesses = read_history(filename_hof)
    
    hof_E = hof_fitnesses[:,E_index] # energy of all history individuals
    hof_CN = hof_fitnesses[:,CN_index] # average CN of all history individuals

    
    
    fig, ax = plt.subplots(figsize= (6,6))
    ax.scatter(-hof_E, -hof_CN, s=50, facecolors='b', edgecolors='k', alpha = 0.5)
    ax.set_xlabel('|Energy| (eV)')
    ax.set_ylabel(r'$\overline{CN}$')
    plt.savefig(os.path.join(os.getcwd(), version, 'Pd'+ str(nseeds) + '_' + str(T) + 'k'  + '_hof.PNG'))
    
    return hof, hof_fitnesses

def plot_best(filename = filename_best, nseeds = nseeds, version = version):
    '''
    This function has been depreciated
    '''
    
    '''
    Read generation best
    '''
    gbest, gbest_fitnesses = read_history(filename_best)
    gbest_E =  gbest_fitnesses[:,E_index] # energy of all best individuals
    gbest_CN = gbest_fitnesses[:,CN_index] # average CN of all history individuals
    gbest_igen = np.arange(1, len(gbest)+1)
    Pdn_atoms, _ = energy.append_support(gbest[-1], super_mother, view_flag=True)


    fig, ax = plt.subplots(figsize= (6,6))
    ax.plot(gbest_igen, gbest_E, 'b--', marker='o', ms=3, markerfacecolor="None", markeredgecolor='b', alpha=1, zorder=0)
    #ax.plot(gbest_igen, gbest_CN,'r--', marker='o', ms=5, markerfacecolor="None", markeredgecolor='r', alpha=1, zorder=0)
    ax.set_xlim(0, max(gbest_igen))
    ax.set_xlabel('# Generation')
    ax.set_ylabel('Energy (eV)')
    plt.savefig(os.path.join(os.getcwd(), version, 'Pd'+ str(nseeds) + '_' + str(T) + 'k' + '_gen_best.PNG'))
    
    return gbest_E, gbest_CN

# Call the functions    
plot_stats()
plot_stats(max_gen = 1000)
hof_inds, hof_fitnesses= plot_hof()

#%% Save the best individuals in the hall of fame
def save_single_POV(atoms, batch_name, index, output_dir):

    pov_args = {
        'transparent': True,  # Makes background transparent. I don't think I've had luck with this option though
        #'run_povray'   : True, # Run povray or just write .pov + .ini files
        'canvas_width': 900,  # Width of canvas in pixels
        #'canvas_height': 500, # Height of canvas in pixels
        'display': False,  # Whether you want to see the image rendering while POV-Ray is running. I've found it annoying
        'rotation': '0x, 0y, -180z',  # Position of camera. If you want different angles, the format is 'ax, by, cz' where a, b, and c are angles in degrees
        # 'rotation': '90x, 0y, -180z', for front views along x axis 
        'celllinewidth': 0.02,  # Thickness of cell lines
        'show_unit_cell': 0  # Whether to show unit cell. 1 and 2 enable it (don't quite remember the difference)
        # You can also color atoms by using the color argument. It should be specified by an list of length N_atoms of tuples of length 3 (for R, B, G)
        # e.g. To color H atoms white and O atoms red in H2O, it'll be:
        #colors: [(0, 0, 0), (0, 0, 0), (1, 0, 0)]
    }

    # Write to POV-Ray file
    filename = str(batch_name) + '_i' + str(index) + '.POV'
    write(os.path.join(output_dir, filename), atoms, **pov_args)
    
#%%
# Select qualified individuals
hof_qualified = []
hof_fitnesses_qualified = []
hof_not_qualified = []
hof_fitnesses_not_qualified = []

for i, (ind_i, fitness) in enumerate(zip(hof_inds, hof_fitnesses)):
    
    # a qualified structure has no floating atoms or isolated atoms
    no_floating_flag = np.abs(fitness[floating_flag_index] - (-1))== 0.0
    no_iso_flag = np.abs(fitness[n_isolate_index]) == 0.0
    
    if no_floating_flag and no_iso_flag:
        hof_qualified.append(ind_i)
        hof_fitnesses_qualified.append(fitness)
    else:
        hof_not_qualified.append(ind_i)
        hof_fitnesses_not_qualified.append(fitness)
        
print('\n{0:6.2f} % individuals qualified\n'.format(len(hof_qualified)/len(hof_inds) *100 ))

# View the atom object for the fittest individual
best_ind = hof_qualified[0]
best_fitness = hof_fitnesses_qualified[0]
best_config = GA_f.one_hot_to_index(best_ind)
best_Pdn_atoms, _ = energy.append_support(best_config, super_mother, view_flag=True)
best_Pdn_atoms = GA_f.arrange_atoms(best_Pdn_atoms)
write(os.path.join(os.getcwd(), version, 'pd' + str(nseeds)+'-opt-CONTCAR'), best_Pdn_atoms)

#%%
# Save top nth into POV
ind_index = list(range(0, np.min([10, len(hof_qualified)]))) 
for i in ind_index:
    ind = hof_qualified[i]
    config_i = GA_f.one_hot_to_index(ind)
    Pdn_atoms, _ = energy.append_support(config_i, super_mother, view_flag=False)
    save_single_POV(Pdn_atoms, 'Pd' + str(nseeds), i, os.path.join(os.getcwd(), version) )
#%%
'''
# View the atom object for an unqualified individual 
i_test = 65
test_ind = hof_inds[i_test]
test_fitness = hof_fitnesses[i_test]
test_config = GA_f.one_hot_to_index(test_ind)
test_atoms, _ = energy.append_support(test_config, super_mother, view_flag=True)   
save_single_POV(test_atoms, 'Pd' + str(nseeds), i_test, os.path.join(os.getcwd(), version) )

#%%
# View some structure in mating
config1 = [ 16,  36,  37,  57, 252, 459, 467, 468, 471, 489, 496, 497, 542, 691, 694, 695, 696, 705, 723, 730]
config2 = [ 47, 228, 229, 231, 232, 233, 234, 235, 236, 238, 239, 248, 450,  451, 452, 476, 481, 686, 715]
config1_mate = [ 16,  36,  37,  57, 252, 459, 467, 468, 471, 489, 496, 497, 542, 691, 694, 452, 476, 481, 686, 730]
config2_mate = [ 47, 228, 229, 231, 232, 233, 234, 235, 236, 238, 239, 248, 450,  451, 695, 696, 705, 723, 715]

configs_mate = [config1, config2, config1_mate, config2_mate]
i_mate = ['c1', 'c2', 'c1_m', 'c2_m']
for ci, config_i in zip(i_mate, configs_mate):
    mate_atoms, _ = energy.append_support(config_i, super_mother, view_flag=True)
    save_single_POV(mate_atoms, 'Pd' + str(nseeds), ci, os.path.join(os.getcwd(), version) )
'''
