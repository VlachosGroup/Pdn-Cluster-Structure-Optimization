# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 00:38:17 2019

@author: yifan
"""

'''
Energy Evaulation function given a super cell
'''
import os
import sys
import json
import pickle
import random

import numpy as np
from ase import Atom, Atoms
from ase.build import surface
from ase.data import covalent_radii
from ase.io import read, write
from ase.visualize import view
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error
from itertools import combinations

import matplotlib
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
font = {'family': 'normal', 'size': 15}
matplotlib.rc('font', **font)

import platform
HomePath = os.path.expanduser('~')
ProjectPath = os.path.join(HomePath, 'Documents', 'GitHub', 
                           'Pdn-Cluster-Structure-Optimization')

# CE Model path
CE_path = os.path.join(ProjectPath, 'lasso-assisted-CE')
sys.path.append(CE_path)

import lattice_functions as lf
from set_ce_lattice import dz, mother

#%%
'''
Useful functions
'''
'''
ind - individal, one hot encoding array consisting of 0s and 1s
ind_index - config or occupied node index list consisting of integer numbers
'''

def occupancy():
    '''
    Creat occupancy for a node in the configuration, value between 0 or 1
    '''
    occ = random.randint(0, 1)
    return occ


def one_hot_to_index(individual):
    '''
    Convert an individual from one hot encoding to a list index
    '''
    ind_index = list(np.nonzero(individual)[0])
    return ind_index


def index_to_one_hot(ind_index, n_nodes):
    '''
    Convert an individual from a list index to one hot encoding
    '''
    individual = np.zeros(n_nodes, dtype=int)
    individual[np.array(ind_index)] = 1

    return individual


def check_Pd_Pd_distance(ind_index, mother):
    '''
    Takes in configuration and return False if atoms are closer than nearest neighbors
    '''

    acceptance_flag = True
    combos = list(combinations(ind_index, 2))
    ncombo = len(combos)

    for i in range(ncombo):
        pt1 = mother[combos[i][0]]
        pt2 = mother[combos[i][1]]
        distance = lf.two_points_D(pt1, pt2)

        if distance < 1.0:
            acceptance_flag = False
            break

    return acceptance_flag


def check_Pd_Pd_neighboring(occ_node_index, ind_index, mother):
    '''
    Takes in a node index and mother
    return if the node is near an existing node
    '''
    acceptance_flag = True
    pt1 = mother[occ_node_index[0]]
    min_distance = np.min([lf.two_points_D(pt1, pt2) for pt2 in mother[ind_index] if not np.all(pt2 == pt1)])
    # print(min_distance)
    if not min_distance == 1.0:
        acceptance_flag = False

    return acceptance_flag


def swap_occ_empty(ind):
    '''
    Core function of the random walk
    Swap an occupied site and an empty site
    takes in one hot numpy array - ind
    return the new configuration and the chosen node
    '''

    x_new = ind.copy()
    occ_indices = np.where(x_new == 1)[0]
    chosen_occ_i = np.random.choice(occ_indices, 1)
    x_new[chosen_occ_i] = 0

    empty_indices = np.where(x_new == 0)[0]
    chosen_empty_i = np.random.choice(empty_indices, 1)
    x_new[chosen_empty_i] = 1

    return x_new, chosen_empty_i, chosen_occ_i


def append_support(ind_index, mother, view_flag=False):
    '''
    Append the configuration onto a ceria support surface
    - Inputs
    - ind_index : the occupied nodes for a given configuration
    - mother : the mother cell
    - view_flag : show in ase GUI
    '''

    # Useful bond information
    Pdr = covalent_radii[46]
    #Or = covalent_radii[8]
    PdPd = Pdr * 2
    PdO = 2.1  # the average PdO length take from CONTCAR files

    def ceria():

        a = 5.49  # Lattice constant
        CeO2 = Atoms('Ce4O8', scaled_positions=[(0., 0., 0.),
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
                     cell=[a, a, a],
                     pbc=True)

        #(1,1,1) is the slab type. There are 2 unit cells along z direction
        slab = surface(CeO2, (1, 1, 1), 2)

        # Repeating the slab 5 unit cells in x and 5 unit cell in y directions
        # At the end the ceria slab is 10 by 10
        # the Pd supercell mother is also 10 by 10
        slab = slab.repeat((5, 5, 1))
        slab.center(vacuum=10.0, axis=2)

        # clave the top layer O atoms
        del slab[[atom.index for atom in slab if atom.z > 15]]

        return slab

    support = ceria()

    # set origin value by looking at the ase GUI, pick one oxygen atom
    origin_index = 17
    origin = support[origin_index].position.copy()
    origin[2] = origin[2] + PdO

    # superpose the Pd lattice onto ceria lattice
    mother_with_support = origin + (mother - mother[0]) * PdPd

    # select the occupied nodes
    Pdpos = mother_with_support[ind_index]

    # append Pd atoms to the support
    nPd = len(Pdpos)
    for i in range(nPd):
        support.append(Atom('Pd', position=Pdpos[i]))

    '''
    Append an atom in the vaccum at the top corner
    for plotting purpose just becase POV is dumb
    '''
    dumb_x = 0  # support.cell[0][0] + support.cell[0][1]
    dumb_y = 0  # support.cell[1][0] + support.cell[1][1]
    dumb_z = support.cell[2][2] - 1
    dumb_pos = np.array([dumb_x, dumb_y, dumb_z])
    support.append(Atom('He', position=dumb_pos))

    if view_flag:
        view(support)

    return support, mother_with_support


def check_floating_atoms(ind, mother):
    '''
    Check if the configuration has any floating atoms in the layer above base layer
    If floating_flag = true, the configuration is considered as infeasible, 
    If floatinfg_flag = false, the configuration can be accepted
    Input the individial one-hot coding and the mother coordinates
    '''
    # Convert to config list
    config = one_hot_to_index(ind)

    # Collect the atoms above the base layer
    config_layer = lf.cal_layers(mother, dz, config)
    config_base_above = list(np.array(config)[np.where(config_layer > 1 )])

    # Check the CN of atoms above the base layer
    Graphs = lf.initialize_graph_object(mother, dz, NN1 = 1)
    Gm = Graphs.Gm
    cn_list = []
    for ci in config_base_above:

        cni = len([i for i in list(Gm.neighbors(ci)) if i in config])
        cn_list.append(cni)

    # Isolated node list, CN < 2
    iso_list = list(np.array(config_base_above)[np.where(np.array(cn_list) < 2)])

    floating_flag = (len(iso_list) > 0)  
    
    return floating_flag

#%%

class Pdn():

    def __init__(self, model_file, mother=mother, super_cell_flag=False):
        '''
        loading the regression results
        '''
        self.mother = mother

        # The flag to inluce 1NN and edges shorter than 1NN
        NN1 = 1
        [self.Gcv, self.J, self.intercept, self.RMSE_test_atom, self.RMSE_test_site] = pickle.load(open(model_file, "rb"))

        self.super_cell_flag = super_cell_flag

        # Initialize graph object
        self.Graphs = lf.initialize_graph_object(self.mother, dz, NN1 = 1)

        # Initialize calculation object

        empty = 'grey'
        filled = 'r'
        occ = [empty, filled]
        self.Cal = lf.calculations(occ)
        self.Gm = self.Graphs.Gm

    def save_super_clusters(self):
        '''
        save the signficant clusters in super cell to a json file
        called 'clusters_super_nonzero.json'
        '''
        with open('clusters_super_cell.json') as f:
            Gcv_super = json.load(f)['Gcv']

        Gcv_super_nonzero = []
        Gcv_model_nonrepeat = [Gi[0] for Gi in self.Gcv]  # take the first clusters in each list

        for Gi_super in Gcv_super:
            for Gi_model_nonrepeat in Gcv_model_nonrepeat:  # check if the first one is in Gcv_super
                if Gi_model_nonrepeat in Gi_super:
                    Gcv_super_nonzero.append(Gi_super)

        # save to a json file
        Gcv_super_nonzero_dict = {'Gcv': Gcv_super_nonzero}
        with open(os.path.join(CE_path, 'clusters_super_nonzero.json'), 'w') as outfile:
            json.dump(Gcv_super_nonzero_dict, outfile)

    def load_super_cluster(self):
        '''
        load 'cluster_super_cell.json'
        '''
        with open(os.path.join(CE_path, 'clusters_super_nonzero.json')) as f:
            self.Gcv_super = json.load(f)['Gcv']
            self.Gcv = self.Gcv_super  # substitue the original Gcv

    def load_config(self, ind_index):
        '''
        load the configuration into self.Graph.Gsv
        '''
        if self.super_cell_flag:
            self.load_super_cluster()
        self.Graphs.get_configs([ind_index])

    def predict_E(self, ind_index):
        '''
        Predict Energy of the cluster only, take in ind index
        '''
        self.load_config(ind_index)
        pi_pred = self.Cal.get_pi_matrix_l(self.Graphs.Gsv, self.Gcv)
        E_pred = float(np.dot(pi_pred, self.J) + self.intercept)

        # return Graphs
        return E_pred, pi_pred

    def swap_occ_empty_fast(self, ind):
        '''
        Core function of the random walk
        Swap an occupied site and a NEARBY (must be 1NN) empty site
        takes in one hot numpy array - ind
        '''
        x_new = ind.copy()
        occ_indices = list(np.where(x_new == 1)[0])
        self.load_config(occ_indices)

        config_G = self.Graphs.Gsv[0]
        NN1_list = []
        for node_i in occ_indices:
            NN1_list += list(config_G.neighbors(node_i))
        NN1_list = list(set(NN1_list))
        NN1_list_empty = [i for i in NN1_list if i not in occ_indices]
                
        chosen_occ_i = np.random.choice(occ_indices, 1)
        chosen_empty_i = np.random.choice(NN1_list_empty, 1)
        
        if not chosen_occ_i ==  chosen_empty_i:
            x_new[chosen_occ_i] = 0
            x_new[chosen_empty_i] = 1

        return x_new, chosen_empty_i, chosen_occ_i
    
    def swap_occ_empty_reverse(self, ind):
        '''
        Core function of the random walk
        Swap an occupied site to an empty site on the base
        takes in one hot numpy array - ind
        '''
        x_new = ind.copy()
        occ_indices = list(np.where(x_new == 1)[0])
        
        base_indices = np.where(self.mother[:,2] == dz)[0]
        base_indices_empty = list(np.where(x_new[base_indices] == 0)[0])

        chosen_occ_i = np.random.choice(occ_indices, 1)
        chosen_empty_i = np.random.choice(base_indices_empty, 1)
        if not chosen_occ_i ==  chosen_empty_i:
            x_new[chosen_occ_i] = 0
            x_new[chosen_empty_i] = 1

        return x_new, chosen_empty_i, chosen_occ_i

    def swap_iso_neighbors(self, ind, alpha1=1.0, alpha2=0.25):
        '''
        New version
        Core function of the random walk
        if there is isolated nodes:
        Randomly put n isolated nodes into NN1 nodes of the existing nodes
        if there is non-isolated nodes:
        Shuffle the occupied nodes to NN1 nodes of the existing nodes

        takes in one hot numpy array - ind
        '''
        x_new = ind.copy()
        config = one_hot_to_index(x_new)  # convert to config

        NN1_list = []  # the NN1 nodes to config
        cn_list = []  # the cn number for each node
        for ci in config:
            NN1_neighbors_i = [i for i in list(self.Gm.neighbors(ci))]
            cni = len([i for i in list(self.Gm.neighbors(ci)) if i in config])
            cn_list.append(cni)
            NN1_list += NN1_neighbors_i
        # Unique NN1 nodes
        NN1_list = list(set(NN1_list))

        # All possible empty NN1 nodes
        NN1_list_empty = [i for i in NN1_list if i not in config]

        # Get both NN1 and NN2 nodes
        NN2_list = []
        for ci in NN1_list:
            NN2_neighbors_i = [i for i in list(self.Gm.neighbors(ci))]
            NN2_list += NN2_neighbors_i

        # Unique NN1 nodes
        NN2_list = list(set(NN2_list + NN1_list))

        # All possible empty NN1 nodes
        NN2_list_empty = [i for i in NN2_list if i not in config]

        # All isolated nodes with coorination number < 2
        iso_list = list(np.array(config)[np.where(np.array(cn_list) < 2)])

        # Given a alpha, determine the number of nodes involved in exchange
        m = int(np.floor(min(len(iso_list), len(NN1_list_empty)) * alpha1))

        if m > 0:  # Randomly put n isolated nodes into NN1 nodes of the existing nodes
            chosen_occ_i = np.unique(np.random.choice(iso_list, m, replace=False))
            x_new[chosen_occ_i] = 0

            chosen_empty_i = np.unique(np.random.choice(NN1_list_empty, m, replace= False))
            x_new[chosen_empty_i] = 1

        if m == 0:  # Shuffle the occupied nodes to NN1 nodes of the existing nodes and choose n from it

            # the number of occupied nodes
            n = len(config)
            n_possible = [n * alpha2, len(NN2_list_empty)]
            if min(n_possible) > 1:
                nswap = int(np.floor(min(n_possible)))
            else: nswap = 1

            #print('\t Swap {} atoms'.format(nswap))
            chosen_occ_i = np.unique(np.random.choice(config, nswap, replace = False))
            x_new[chosen_occ_i] = 0

            chosen_empty_i = np.unique(np.random.choice(NN2_list_empty, nswap, replace= False))
            x_new[chosen_empty_i] = 1

        return x_new, chosen_empty_i, chosen_occ_i
