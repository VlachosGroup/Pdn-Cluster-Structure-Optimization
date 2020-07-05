# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 08:57:56 2020

@author: Yifan Wang

Script to analysis GA trajectory
    
"""

#%% Import session

import os
import sys
import pandas as pd
import numpy as np
from ase.io import read, write

import platform
HomePath = os.path.expanduser('~')
ProjectPath = os.path.join(HomePath, 'Documents', 'GitHub', 
                           'Pdn-Cluster-Structure-Optimization')

# Energy model directory
energy_path = os.path.join(ProjectPath, 'lasso-assisted-CE')
sys.path.append(energy_path)

import energy_functions as energy

# LASSO model directory
selected_batches = [0, 1, 2, 3]
lasso_model_name = 'lasso' + '_' + ''.join(str(i) for i in selected_batches)
lasso_path = os.path.join(energy_path, lasso_model_name)
lasso_file = os.path.join(lasso_path, lasso_model_name + '.p')



from generate_clusters_super_cell import super_mother
from set_ce_lattice import dz

import GA_functions as GA_f
import kinetics_functions as kinetics


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
    
    # Convert the pandas value to one hot encoding
    population = []
    for indi in df['individual']:
        population.append([int(i) for i in indi.strip('][').split(', ')])
    
    # Save the fitness value into a 2D array
    fitnesses = np.column_stack((df['fitness1'], df['fitness2'], df['fitness3'], df['fitness4'], df['fitness5']))
    
    # Generation number in a 1D array
    generation_no = np.array(df['generation no.'])
    
    return population, fitnesses, generation_no

    
#%% Simulation specifications
T = 300  # Ensemble temperature
nseeds = 20 # Ensemble size
GA = GA_f.generator(lasso_file, mother=super_mother, super_cell_flag=True, T=T, nseeds=nseeds)

ncutoff_0 = GA.ncutoff_0
ncutoff_1 = GA.ncutoff_1 # the cutoff between small and large clusters
ncutoff_2 = GA.ncutoff_2

if nseeds < ncutoff_0: E_index, CN_index = 3, 4
if (nseeds >= ncutoff_0) and (nseeds < ncutoff_1): E_index, CN_index = 2, 4
if nseeds >= ncutoff_1 and nseeds < ncutoff_2: E_index, CN_index = 1, 3
if nseeds >= ncutoff_2: E_index, CN_index = 2, 4 #4, 3 

# The number of Pd atoms in the ensemble
version_list = ['pd5', 'pd10', 'pd15', 'pd20_s0', 'pd20_s1', 'pd20_s2', 'pd20_800k', 'pd20_1300k', 'pd25', 'pd30']
version= 'pd20_traj' #version_list[5]
filename_history = os.path.join(os.getcwd(), version, 'ga_history_output_' + str(nseeds) + '_' + str(T) + 'k' + '.csv')

#%% Plot and save the results
hof_inds, hof_fitnesses, generation_no = read_history(filename_history)

#%%

def plot_descriptor(y,  descriptor_name):
    """Input the descriptor name and plot it
    
    :param descriptor: 'GCN', 'CN1', 'CN2', 'surface atom ratio', defaults to 'CN1'
    :type descriptor: str, optional
    :param descriptor: x values on the plot
    :type descriptor: numpy array, optional
    
    :return: fig 
    :rtype: the figure handle
    """
    

    # Calculate the mean and variance, making sure yi is an empty array
    y_means = np.array([np.mean(yi) for yi in y if len(yi) > 0])
    y_std = np.array([np.std(yi) for yi in y if len(yi) > 0])
    
    
    y_max = np.array([np.max(yi) for yi in y if len(yi) > 0]) #y_means + y_std*1.98 #
    y_min =np.array([np.min(yi) for yi in y if len(yi) > 0]) # y_means - y_std*1.98 #
    
    #ymax_plot = np.ceil(np.max(y_max)*1.1)
    #if descriptor_name == 'surface atom ratio': ymax_plot = 1.1
    
    max_gen = len(y_means)
    x = np.arange(0, max_gen)
    
    # Set the x values as indices

    fig, ax = plt.subplots(figsize= (5,5))
    ax.plot(x, y_min, 'steelblue', linewidth = 0.8)
    ax.plot(x, y_max, 'steelblue', linewidth = 0.8)
    ax.plot(x, y_means, 'k', label = 'Mean')
    ax.fill_between(x, y_min, y_max, color='steelblue', alpha=0.3, label = 'Range')
    ax.set_xlabel('Generation no.')
    ax.set_ylabel(descriptor_name)
    
    #ax.set_ylim(0, ymax_plot)
    #ax.legend(frameon=False)
    
    if max_gen > 1000:
    # Set the log scale
        ax.set_xscale('log')
        ax.set_xlim(1, max_gen)
        ax.set_xticks([1,10,100,1000,max_gen])
    if descriptor_name in ['GCN', 'CN1', 'CN2']:
        ax.set_ylim(0, 10)
    if descriptor_name == 'surface atom ratio':
        ax.set_ylim(0.5, 1.05)
        ax.set_ylabel('Surface Atom Fraction')

    #plt.tight_layout()
    return fig
    
    
def get_descriptor_plots(trajectory, OutputPath):
     """ plot all descriptors and save the plot to PNGs
     """
     descriptors = ['GCN', 'CN1', 'CN2', 'surface atom ratio']
     for di in descriptors:
         di_list = trajectory.read_descriptors(di)
         fig = plot_descriptor(di_list, di)
         fig.savefig(os.path.join(OutputPath, di + '_trajectory.PNG'), dpi = dpi)
         
         
def get_metrics_list(hof_fitnesses, generation_no):
    
    max_gen = int(np.max(generation_no))
    generation_no_list = np.arange(0, max_gen)
    metrics_list = []  #2D lists
    for mi in range(hof_fitnesses.shape[1]):
        metrics_list.append([])
        for gi in generation_no_list:
            fitnesses_gen_i = hof_fitnesses[np.where(generation_no == gi)][:,mi]
            metrics_list[mi].append(fitnesses_gen_i)
    return metrics_list
    
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
CurrentPath = os.path.dirname(os.path.realpath(__file__))
OutputPath = os.path.join(CurrentPath, version)
mother = super_mother

# Create a trajectory object
trajectory = kinetics.Trajectory(mother, dz, CurrentPath)
trajectory.get_configs(hof_inds, generation_no)
#trajectory.get_descriptors(generation_no)


#%% Plot the descriptors
#CN1_list = trajectory.read_descriptors('CN1')
#plot_descriptor(CN1_list, 'CN1')

get_descriptor_plots(trajectory, OutputPath)


#%% Plot the GCA metrics
metrics_list = get_metrics_list(hof_fitnesses, generation_no)
metrics_names = ['floating_flag', 'Energy (eV)', r'$\rm n_{u}$', r'$\rm -\overline{CN1}$', r'$\rm \overline{Z}$']

m_interested = [1,2,3,4]
for mi in m_interested:
    plot_descriptor(metrics_list[mi], metrics_names[mi])    
    
#%%
# Save top nth into POV
max_gen = int(np.max(generation_no))
ind_index = list(range(0, 5)) + list(range(max_gen-5, max_gen))
for i in ind_index:
    ind = hof_inds[i]
    config_i = GA_f.one_hot_to_index(ind)
    Pdn_atoms, _ = energy.append_support(config_i, super_mother, view_flag=False)
    save_single_POV(Pdn_atoms, 'Pd' + str(nseeds), i, OutputPath)
    