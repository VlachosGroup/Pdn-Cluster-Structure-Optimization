# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:55:36 2018

@author: wangyf
"""

import sys
import os
import numpy as np

import platform
HomePath = os.path.expanduser('~')
ProjectPath = os.path.join(HomePath, 'Documents', 'GitHub', 
                           'Pdn-Cluster-Structure-Optimization')

# Energy model directory
energy_path = os.path.join(ProjectPath, 'lasso-assisted-CE')
sys.path.append(energy_path)

import energy_functions as energy

from generate_clusters_super_cell import super_mother
from set_ce_lattice import dz


import lattice_functions as lf


def inverse_ab(ab):
    '''
    Input a tuple consisting of two elements
    and return a tuple with elements flipped
    '''
    a = ab[0]
    b = ab[1]
    return (b, a)


'''
Create an empty cluster object
'''
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
draw = [0, 0, 0]


gm_graph = lf.graphs(occ, NN1, draw)
gm_graph.get_mother(super_mother, dz)
# obtain the mother graph
Gm = gm_graph.Gm

#%%  Method 1
# Get the 1st nearest neighbor edges
NN1_edges1 = list(Gm.edges)

# Count the flipped egdes
NN1_edges2 = []
for edge in NN1_edges1:
    NN1_edges2.append(inverse_ab(edge))

# Concatenate the two
NN1_edges = NN1_edges1 + NN1_edges2


def structure_ids_v1(occ_nodes):
    '''
    Generate metrics for a given structure
    '''

    count = 0
    connected_edges = []

    for i, nodei in enumerate(occ_nodes):
        nNN1 = 0
        edges = []
        for j, nodej in enumerate(occ_nodes):
            if not nodei == nodej:
                edges.append((nodei, nodej))
        for edge in edges:
            if edge in NN1_edges:
                nNN1 = nNN1 + 1
                connected_edges.append(edge)

        if nNN1 >= 3:
            count = count + 1

    connected_edges_array = np.array(connected_edges)
    repeated_nodes_array = np.reshape(connected_edges_array, connected_edges_array.size)
    unique_nodes, node_edge_counts = np.unique(repeated_nodes_array, return_counts=True)
    node_edge_counts = node_edge_counts / 2
    # each node should have more than two edges at the same time
    # nodes indices with more than 2 edges
    edge2_nodes = np.where(node_edge_counts >= 2)[0]
    # nodes indices with more than 3 edges
    edge3_nodes = np.where(node_edge_counts >= 3)[0]
    # nodes indices with more than 4 edges
    edge4_nodes = np.where(node_edge_counts >= 4)[0]

    # nodes with less than 2 edges, need to be minimized
    n1_nodes = len(occ_nodes) - len(edge2_nodes)
    # nodes with less than 3 edges, need to be minimized
    n2_nodes = len(occ_nodes) - len(edge3_nodes)
    # nodes with less than 4 edges, need to be minimized
    n3_nodes = len(occ_nodes) - len(edge4_nodes)

#    if edges == []: score = 0
#    else: score = 1- count/len(edges)
    # print(nodes_counts)

    '''
    score based on where atom is in the cluster
    '''
    nodes_array = np.array(occ_nodes)
    nl1 = len(np.where(super_mother[nodes_array][:, 2] == 1)[0])
    nl2 = len(np.where(super_mother[nodes_array][:, 2] == 2)[0])
    nl3 = len(np.where(super_mother[nodes_array][:, 2] == 3)[0])
    nl4 = len(np.where(super_mother[nodes_array][:, 2] == 4)[0])

    # Calculate the average layer number
    # (avergae center of mass from the support)
    layer_average = np.dot(np.array([1, 2, 3, 4]), np.array([nl1, nl2, nl3, nl4]))

    return n1_nodes, n2_nodes, n3_nodes, layer_average

#%% Method 2


def structure_ids(config):
    '''
    Input the occupied nodes in a structure - config
    Returns the  metrics evaluating the structure
    '''
    # The list to store 1NN coordination number for each node
    cn = []

    for ci in config:
        # 1NN neighbors in the structure
        NN1_neighbors = [i for i in list(Gm.neighbors(ci)) if i in config]
        cni = len(NN1_neighbors)
        cn.append(cni)

    # The average 1NN coordination number
    # Negation due to miniziation purpose
    cn_average_neg = - np.mean(cn)

    # The number of isolated single atoms
    # node with cn <= 1, one or no edge, isolated single nodes
    # this should be minimized
    n_isolate = len(np.where(np.array(cn) < 2)[0])
    # The node with cn < 3, should be minimized
    cn3_below = len(np.where(np.array(cn) < 3)[0])
    # The node with cn < 4, one edge
    cn4_below = len(np.where(np.array(cn) < 4)[0])

    # score based on where atom is in the cluster

    nodes_array = np.array(config)
    nl1 = len(np.where(super_mother[nodes_array][:, 2] / dz == 1)[0])
    nl2 = len(np.where(super_mother[nodes_array][:, 2] / dz == 2)[0])
    nl3 = len(np.where(super_mother[nodes_array][:, 2] / dz == 3)[0])
    nl4 = len(np.where(super_mother[nodes_array][:, 2] / dz == 4)[0])

    # Calculate the average layer number
    # (avergae center of mass from the support)
    layer_average = np.dot(np.array([1, 2, 3, 4]),
                           np.array([nl1, nl2, nl3, nl4])) / len(config)

    # check if there is atoms in the best layer
    # if yes return -1, if not return 0
    if nl1 > 0:
        floating_flag = -1
    else:
        floating_flag = 0

    return n_isolate,  floating_flag , cn_average_neg, layer_average


'''
Test the method
'''
if __name__ == "__main__":

    config = [0, 1, 5, 51]  # testing configuration

    n_isolate,  floating_flag , cn_average_neg, layer_average = structure_ids(config)

    print("The structure ID: {} isolated atoms, {} cn average,  {} layer average"
          .format(n_isolate,  cn_average_neg, layer_average))
