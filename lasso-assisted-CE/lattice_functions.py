# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:08:30 2018

@author: wangyf
"""
'''
Objective Oriented Version of Lattice Building functon
'''


import numpy as np
import math
import json
import networkx as nx
from networkx.algorithms import isomorphism as iso
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%
'''
Basic functions
'''

def two_points_D(A,B):
    
    '''
    Calculates the distance between two points A and B
    '''
    n = len(A)
    s = 0
    for i in range(n):
        s = s+ (A[i]-B[i])**2
    d = math.sqrt(s)
    d = float(format(d, ' .3f')) # round off to three decimal digits
    return d

def two_points_D_np(A, B):
    
    '''
    np norm method
    '''
    A = np.array(A)
    B = np.array(B)
    
    d = np.linalg.norm(A-B)
    
    return d


def drawing(G):
    '''
    takes in graph g, draw it in 2D
    '''
    color = nx.get_node_attributes(G,'color')
    pos= nx.get_node_attributes(G,'pos')
    plt.figure() 
    nx.draw(G, pos, with_labels=False, node_color = list(color.values()))
    return plt

def drawing3D(G, pos, sitetype,  cell = []):
    '''
    Draw the 3D graph and color code the occupied 
    based on different site types
    '''
    cart_coords_3d = pos.copy()
    neighbor_list = list(G.edges)
    node_colors = list(nx.get_node_attributes(G,'color').values())
    #nodes_layers = list(nx.get_node_attributes(G,'z').values())
    
    plot_neighbs = True
    y_rotate = -30
    z_rotate = 10
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_aspect('equal')

    if plot_neighbs:
        for pair in neighbor_list:                                             # neighbors
            p1 = cart_coords_3d[pair[0]]
            p2 = cart_coords_3d[pair[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], '--k', linewidth=0.05)
    
    for i, point_i in enumerate(cart_coords_3d):
        
        if node_colors[i] == 'grey':
            point_color = 'grey'
            ax.scatter(point_i[0], point_i[1], point_i[2],  marker= 'o' , color=point_color, s= 30, alpha = 0.5)  
            
        if node_colors[i] == 'r': 

            if sitetype[i] == 'a' or sitetype[i] == 'top': point_color = 'green'
            elif sitetype[i] == 'c' or sitetype[i] =='bridge': point_color = 'darkorange'
            elif sitetype[i] == 'b' or sitetype[i] == 'hollow': point_color = 'blue'
            ax.scatter(point_i[0], point_i[1], point_i[2],  marker= 'o' , color=point_color, s= 100, edgecolors = 'k', alpha = 0.9)  

    '''
    Set axis in equal scale
    '''
    
    X_scatter = cart_coords_3d[:,0]
    Y_scatter = cart_coords_3d[:,1]
    Z_scatter = cart_coords_3d[:,2]

    max_range = np.array([X_scatter.max()-X_scatter.min(), Y_scatter.max()-Y_scatter.min(), Z_scatter.max()-Z_scatter.min()]).max()


    ax.set_xlim(X_scatter.min(), X_scatter.min() + max_range)
    ax.set_ylim(Y_scatter.min(), Y_scatter.min() + max_range)
    ax.set_zlim(Z_scatter.min(), Z_scatter.min() + max_range)
    ax.axis('off')

    ax.view_init(z_rotate, y_rotate)

    # Add the cell vector
    if not cell == []:
        
        ax = plot_3D_box(ax, pos, cell)
        
    plt.tight_layout()

    return fig, ax

    
def LeaveOneOut(A, a):
    '''
    takes in a list A and returns a new list B by leaving ath element out
    '''     
    B = [x for i,x in enumerate(A) if i!=a]
    
    return B 

def add_z(v, z):
    '''
    takes in a np array and add z coordinates to it
    '''
    vd = np.concatenate((v, np.array([z*np.ones(len(v))]).T), axis =1) 

    return vd


def cal_layers(mother, dz, config):

    '''
    takes in mother coordinates and one configurations and 
    calculate how the layer number each atom is in
    '''
    config_layers = np.around(mother[np.array(config)][:,2]/dz, decimals = 0).astype(int)
    
    return config_layers

def get_layers(mother, dz, config):

    '''
    takes in mother coordinates and one configurations and 
    returns how the number of layers there are
    '''
    config_layers = np.around(mother[np.array(config)][:,2]/dz, decimals = 0).astype(int)
    n_layers = np.amax(config_layers)
    
    return n_layers

def get_node_layer_dict(mother, dz):
    '''
    takes in mother coordinates and dz
    returns the a dictionary containing which nodes in which layer
    '''
    
    node_layer_v = np.around(mother[:,2]/dz, decimals = 0).astype(int)
    node_layer_dict = dict()
    n_layer = np.amax(node_layer_v)
    
    for layer_i in range(n_layer):
        node_layer_dict[layer_i] = list(np.where(node_layer_v == layer_i +1)[0])  
        
    return node_layer_dict


def get_NPd_list(config_list):
    
    '''
    takes in configuration list 
    '''
    # the number of Pd atoms in each structure
    NPd_list = np.array([len(x) for x in config_list])
    
    return NPd_list


#%%
'''
Construct periodic boundary conditions from ase package
Contains one cube and one fcc (ABC) structure
'''
from ase.build import bcc100, fcc111


def plot_3D_box(ax, pos, cell):

    unit_points = np.array([[0, 0, 0],
                             [1, 0, 0],
                             [1, 1, 0],
                             [0, 1, 0],
                             [0, 0, 1],
                             [1, 0, 1 ],
                             [1, 1, 1],
                             [0, 1, 1]])
    Z = np.zeros((8,3))
    for i in range(len(Z)): Z[i,:] = np.dot(unit_points[i,:], cell)
    Z[:,2] = Z[:,2] + np.min(pos[:,2]) # Shift the cell to the base layer in z direction
    for i in range(0, len(Z)):
        ax.scatter(Z[i,0], Z[i, 1], Z[i, 2], color = 'k')
        #ax.text(Z[i,0], Z[i, 1], Z[i, 2], '%s' % (str(i)) )
        
    edges = [[0,1],[1,2],[2,3],[0,3],
             [0,4],[1,5],[2,6],[3,7],
             [4,5],[5,6],[6,7],[7,4]]
    
    for ei in edges:
        ax.plot(Z[ei,0], Z[ei,1], Z[ei,2], 'k--')
    
    return ax

## Make a plot function for the lattice
def plot_CE_lattice(pos, view_lables = False, cell = [], initial_index = 0, input_color = 'lightgrey'):
    '''
    plot the cluster expansion lattice 
    and draw dash line for the cell vector
    '''
    cart_coords_3d = pos
    
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')
    y_rotate = -30
    z_rotate = 10
    
    point_color = input_color
    label_color = input_color
    
    
    for i, point_i in enumerate(cart_coords_3d): 
        ax.scatter(point_i[0], point_i[1], point_i[2],  marker= 'o' , color=point_color, s= 40, edgecolors = 'k', alpha = 0.9)  
        if view_lables:
            ax.text(point_i[0], point_i[1], point_i[2], '%s' % (str(i + initial_index)), size=10, zorder=1, color=label_color)
    '''
    Set axis in equal scale
    '''
    
    X_scatter = cart_coords_3d[:,0]
    Y_scatter = cart_coords_3d[:,1]
    Z_scatter = cart_coords_3d[:,2]

    max_range = np.array([X_scatter.max()-X_scatter.min(), Y_scatter.max()-Y_scatter.min(), Z_scatter.max()-Z_scatter.min()]).max()


    ax.set_xlim(X_scatter.min(), X_scatter.min() + max_range)
    ax.set_ylim(Y_scatter.min(), Y_scatter.min() + max_range)
    ax.set_zlim(Z_scatter.min(), Z_scatter.min() + max_range)
    ax.axis('off')
    ax.view_init(z_rotate, y_rotate)
    
    # Add the cell vector
    if not cell == []:
        
        ax = plot_3D_box(ax, pos, cell)
        
    plt.tight_layout()
    plt.show()
 
    
def drawing3D_super(G, pos, sitetype, cell = []):
    
    '''
    For super cell, larger graph 
    Draw the dash line for cell vector
    Draw the 3D graph and color code the occupied 
    based on different site types
    '''
    cart_coords_3d = pos.copy()
    neighbor_list = list(G.edges)
    node_colors = list(nx.get_node_attributes(G,'color').values())
    #nodes_layers = list(nx.get_node_attributes(G,'z').values())
    
    plot_neighbs = True
    y_rotate = -30
    z_rotate = 10
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_aspect('equal')

    if plot_neighbs:
        for pair in neighbor_list:                                             # neighbors
            p1 = cart_coords_3d[pair[0]]
            p2 = cart_coords_3d[pair[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], '--k', linewidth=0.05)
    
    for i, point_i in enumerate(cart_coords_3d):
        
        if node_colors[i] == 'grey':
            point_color = 'grey'
            ax.scatter(point_i[0], point_i[1], point_i[2],  marker= 'o' , color=point_color, s= 10, alpha = 0.5)  
            
        if node_colors[i] == 'r': 

            if sitetype[i] == 'a': point_color = 'green'
            elif sitetype[i] == 'c': point_color = 'darkorange'
            elif sitetype[i] == 'b': point_color = 'blue'
            ax.scatter(point_i[0], point_i[1], point_i[2],  marker= 'o' , color=point_color, s= 100, edgecolors = 'k', alpha = 0.9)  

    '''
    Set axis in equal scale
    '''
    
    X_scatter = cart_coords_3d[:,0]
    Y_scatter = cart_coords_3d[:,1]
    Z_scatter = cart_coords_3d[:,2]

    max_range = np.array([X_scatter.max()-X_scatter.min(), Y_scatter.max()-Y_scatter.min(), Z_scatter.max()-Z_scatter.min()]).max()


    ax.set_xlim(X_scatter.min(), X_scatter.min() + max_range)
    ax.set_ylim(Y_scatter.min(), Y_scatter.min() + max_range)
    ax.set_zlim(Z_scatter.min(), Z_scatter.min() + max_range)
    ax.axis('off')

    ax.view_init(z_rotate, y_rotate)

    # Add the cell vector
    if not cell == []:
        
        ax = plot_3D_box(ax, pos, cell)
        
    plt.tight_layout()

    return fig, ax


    
    
    
def plot_CE_layers(pos, view_lables):
    '''
    Plot one layer per graph for a clutser expansion lattice
    '''
    z_pos = pos[:,2]
    z_values = np.unique(z_pos)

    pos_layer = []
    pos_numbers = [0]
    for zi in z_values:
        # get the 2d position of layer a, b and c
        pos_layer.append(pos[np.where(z_pos == zi)])
        pos_numbers.append(len(pos[np.where(z_pos == zi)]))
        
    pos_cum_numbers = np.cumsum(np.array(pos_numbers))
    
    for pos_layer_i, initial_i in zip(pos_layer, pos_cum_numbers):
        plot_CE_lattice(pos_layer_i, view_lables, initial_index = initial_i)

def build_pbc_cube(size_v = (1,1), dz = 1, lattice_c = 1, view_flag = False, view_labels = False):
    '''
    build 3D periodic cluster expansion lattice, only 2 layers in z direction
    example: build_pbc_cube((2,2)) to create a 2 by 2 unite lattice
    '''
    size_x = size_v[0]
    size_y = size_v[1]
    size_z = 2
    element = 'He' # space-filling element
    # use the base layer of bcc100 plane
    layer1 = bcc100(element, size=[size_x, size_y, 1], a = lattice_c)
    pos1 = layer1.get_positions()
    pos2 = layer1.get_positions()
    pos2[:,2] = dz 
    
    # get the position of all lattice points
    pos_all = np.concatenate((pos1, pos2) ,axis = 0)
    
    # get the 3d cell vector
    cell_all = layer1.get_cell() # get the total cell size 
    cell_all[2:,2] = size_z
    
    if view_flag:
        plot_CE_lattice(pos_all,  view_labels, cell = cell_all)
        
    return pos_all, cell_all
    


def build_pbc_fcc_abc(size_v = (1,1), layer_structure = ['a','bc','abc', 'abc'], lattice_c = 1, view_flag = False, view_labels = False):
   
    '''
    build 3D periodic cluster expansion lattice
    z direction follows the abc rules
    example: build_pbc_fcc_abc((3,3)) to create a 3 by 3 unit lattice
    '''
    
    size_x = size_v[0]
    size_y = size_v[1]
    size_z = len(layer_structure)
    
    element = 'H' # space-filling element
    # use 3 layer fcc structure
    # somehow a is not equal to , we have to scale it
    s_factor = lattice_c/0.57735027* (6**0.5/3)
    layers_abc = fcc111(element, size=[size_x, size_y, 3], a = s_factor* lattice_c ) 
    
    pos_abc = layers_abc.get_positions()
    
    # get the z values in fcc lattice and get the average dz
    z_pos = pos_abc[:,2]
    z_values = np.unique(z_pos) 
    dz =  np.mean(z_values[1:] - z_values[:-1]) #6**0.5/3 #
    
    # get the 2d position of layer a, b and c
    pos_a = pos_abc[np.where(z_pos == z_values[0])][:,0:2]
    pos_b = pos_abc[np.where(z_pos == z_values[2])][:,0:2]
    pos_c = pos_abc[np.where(z_pos == z_values[1])][:,0:2]
    
    
    # construct the lattice based on input layer structure and save to pos_layer 
    pos_list = []
    type_list = []
    for layer_i, layer_si in enumerate(layer_structure):
        
        pos_layer = []
        type_layer = []
        zi = dz * (layer_i+1)
        
        for ci in list(layer_si):
            if ci == 'a': pos_layer.append(add_z(pos_a, zi))
            if ci == 'b': pos_layer.append(add_z(pos_b, zi))
            if ci == 'c': pos_layer.append(add_z(pos_c, zi))
            type_layer += list(np.repeat(ci, size_x*size_y))
            
        pos_list.append(np.concatenate(pos_layer))
        type_list += type_layer
        
    # concatenate all layers together
    pos_all = np.concatenate(pos_list)
    type_all = type_list.copy()

    # get 3d cell vector
    cell_all = layers_abc.get_cell()
    cell_all[2:,2] = np.max(pos_all[:,2])       
        
    if view_flag:
        plot_CE_lattice(pos_all, view_labels, cell = cell_all)

    return pos_all, cell_all, type_all


#%%
'''
The main object for cluster expansion to create network graphs
generate graphs for both configurations and clusters
'''

class graphs():
    
    def __init__(self, occupancy, NN1, draw):
        
        '''
        takes in the occupancy color vector 
        occupancy[0] is the empty color
        occupancy[1] is the filled color
        '''
        
        self.occupancy = occupancy
        self.empty = self.occupancy[0]
        self.filled = self.occupancy[1]
        self.NN1 = NN1
        self.draw = draw
        
    def gmothers(self, mother, dz):
    
        '''
        takes in mother cooridate list 
        returns connected lattice graph
        '''
        draw_mother = self.draw[0]
        self.mother = mother
        self.nm = len(mother)
        self.dz = dz
        Gm = nx.Graph()
        
        for i in range(self.nm):
            Gm.add_node(i, pos = mother[i][:2], z = str(int(mother[i][2]/self.dz)), color = self.empty)
        
        
        self.edge = []
        self.edge_d = []
        self.edge_z = []
        
        # Add all egdes and calculate the edge distance
        for i in range(self.nm):
            for j in np.arange(i+1,self.nm):
                self.edge.append((i,j))
                self.edge_d.append(two_points_D(mother[i],mother[j]))
                self.edge_z.append(str(int(mother[i][2]/self.dz))+str(int(mother[j][2]/self.dz)))
                
                
        self.ne = len(self.edge)
        for i in range(self.ne): 
            
            if self.NN1: # only draw 1st Nearest Neighbors 
                if self.edge_d[i] == 1.0: 
                    Gm.add_edges_from([self.edge[i]], z = self.edge_z[i], length = self.edge_d[i])
            
            # if self.NN1 == 2: # draw both 1NN and edges shorter than 1NN    
            #     if self.edge_d[i] <= 1.0: 
            #         Gm.add_edges_from([self.edge[i]], z = self.edge_z[i], length = self.edge_d[i])
            
            else:
                Gm.add_edges_from([self.edge[i]],  z = self.edge_z[i], length = self.edge_d[i])
        
        if draw_mother:
            drawing(Gm)
            plt.title('%d lattice points' %self.nm)
            
        return Gm
    
    
    def gconfigurations(self, son):
        
        '''
        takes in mother coordinate list and son's index number and occupancy vector
        returns the shaded son graph
        '''   
        draw_config = self.draw[1]
        ns = len(son)
        Gs = nx.Graph()

        for i in range(self.nm):
            Gs.add_node(i, pos = self.mother[i][:2], z = str(int(self.mother[i][2]/self.dz)), color = self.empty)

        for i in range(self.ne): 
            
            if self.NN1: # only draw 1st Nearest Neighbors 
                if self.edge_d[i] == 1.0:
                    Gs.add_edges_from([self.edge[i]], z = self.edge_z[i], length = self.edge_d[i])
            
            # if self.NN1 == 2: # draw both 1NN and edges shorter than 1NN    
            #     if self.edge_d[i] <= 1.0: 
            #         Gs.add_edges_from([self.edge[i]], z = self.edge_z[i], length = self.edge_d[i])
            
            else:
                Gs.add_edges_from([self.edge[i]], z = self.edge_z[i], length = self.edge_d[i])

        for si in range(ns):
            Gs.nodes[son[si]]['color'] = self.filled
        
        if draw_config:
            drawing(Gs)
            plt.title('Pd %d' %ns)
        
        return Gs
    

    def gclusters(self, cmother, cson, cNN1 = False):
    
        '''
        takes in clusters 
        cmother are coordinates
        return cluster graph objective
        '''
        draw_clusters = self.draw[2]
        Gc = nx.Graph()
        cns = len(cson)
        
        for i in range(cns):
            c = cson[i]
            Gc.add_node(i, pos = cmother[c][:2], z = str(int(cmother[c][2]/self.dz)), color = self.filled)
            
        cedge = []
        cedge_d = [] # the distance of each edge
        cedge_z = [] # the layer number of each edge
        
        for i in range(cns):        
            for j in np.arange(i+1,cns):
                c  = cson[i]
                d = cson[j]
                cedge.append((i,j))
                cedge_d.append(two_points_D(cmother[c],cmother[d])) 
                cedge_z.append(str(int(cmother[c][2]/self.dz))+str(int(cmother[d][2]/self.dz)))
        
        cne = len(cedge)
        for i in range(cne):
            
            if cNN1: # only include neighboring edges in the graph
                if cedge_d[i] == 1.0: 
                    Gc.add_edges_from([cedge[i]], z = cedge_z[i], length = cedge_d[i])
            
            # if cNN1 == 2: # draw both 1NN and edges shorter than 1NN    
            #     if self.edge_d[i] <= 1.0: 
            #         Gc.add_edges_from([self.edge[i]], z = self.edge_z[i], length = self.edge_d[i])
            
            else:
               Gc.add_edges_from([cedge[i]], z = cedge_z[i], length = cedge_d[i])
               
        if draw_clusters:            
            drawing(Gc)
            plt.title('Pd %d' %cns)
            
        return Gc
    

    def gclusters_kmc(self, Gm_NN1, cmother, cson, cNN1 = False):
    
        '''
        takes in clusters 
        cmother are coordinates
        return cluster graph objective
        '''
        draw_clusters = self.draw[2]
        Gc = nx.Graph()
        cns = len(cson)
        
        for i in range(cns):
            c = cson[i]
            Gc.add_node(i, pos = cmother[c][:2], z = str(int(cmother[c][2]/self.dz)), color = self.filled)
            
        cedge = []
        cedge_d = [] # the distance of each edge
        cedge_z = [] # the layer number of each edge
        
        for i in range(cns):        
            for j in np.arange(i+1,cns):
                c  = cson[i]
                d = cson[j]
                cedge.append((i,j))
                cedge_d.append(two_points_D(cmother[c],cmother[d])) 
                cedge_z.append(str(int(cmother[c][2]/self.dz))+str(int(cmother[d][2]/self.dz)))
        
        cne = len(cedge)
        for i in range(cne):
            if cNN1: # only include neighboring edges in the graph
                if cedge_d[i] <= 1: 
                    Gc.add_edges_from([cedge[i]], z = cedge_z[i], length = cedge_d[i])
            else:
               Gc.add_edges_from([cedge[i]], z = cedge_z[i], length = cedge_d[i])
               
        if draw_clusters:            
            drawing(Gc)
            plt.title('Pd %d' %cns)
            
        return Gc
    

    def get_mother(self, mother, dz):
        '''
        takes in mother coordinates list and 
        add mother attribute to the class
        '''
        
        self.Gm  = self.gmothers(mother, dz)
        
        
    def get_configs(self, config):
        
        '''
        takes in configuration index list
        get a list of configurations as graph objects
        '''
        
        self.Gsv = []
        self.nconfig = len(config)
        
        for si in range(self.nconfig):
            son_i = config[si]
            Gs = self.gconfigurations(son_i)
            self.Gsv.append(Gs)
     
        
    def get_clusters(self, cmother, ccluster, cNN1 = False):
        
        '''
        takes in cluster coordinates list and cluster index list
        returns a list of clusters as graph objects
        '''
        
        self.nc = len(ccluster) # number of clusers
        self.Gcv = [] # list of clusters
        
        for si in range(self.nc):
            cson = ccluster[si] 
            Gc = self.gclusters(cmother,cson, cNN1)
            self.Gcv.append(Gc)   
            
           

#%%
'''
Cluster related functions
'''
      
def initialize_graph_object(mother, dz, NN1 = 1):

    '''
    NN1 == 0, draw all nodes
    NN1 == 1, only draw 1st nearest neighbors
    NN1 == 2, connect both 1st nearest neighbors 
    and edges with length smaller than 1NN
    '''


    '''
    Initialize graph object function
    '''
    empty = 'grey'
    filled = 'r'
    occ = [empty, filled]
    

    '''
    Draw mother/conifgurations/clusters?
    '''
    draw = [0, 0, 0]
    
    
    Graphs = graphs(occ, NN1, draw)
    Graphs.get_mother(mother, dz)
    
    return Graphs


    
#%%        
class calculations():
    
    '''
    Perform statistical calculation for cluster expansion
    '''
    
    def __init__(self,occupancy):
        
        '''
        takes in the occupancy color vector 
        occupancy[0] is the empty color
        occupancy[1] is the filled color
        '''
        
        self.occupancy = occupancy
        self.empty = self.occupancy[0]
        self.filled = self.occupancy[1]
        
        
    def get_occupancy(self, G, i):
        
        '''
        Get the occupancy from the graph G for node i 
        Occupied is 1 and unoccupied is 0, ###!!! is it 0 or 1???
        '''
        
        if G.nodes[i]['color'] == self.empty: o = 0
        if G.nodes[i]['color'] == self.filled: o = 1 
        
        return o    

    def get_delta_G(self, Gl, Gs):
        
        '''
        This code might be problematic
        takes in larger graph Gl and smaller graph Gs
        find sub isomorphic graphs of Gs from Gl
        calculate the delta value in pi matrix 
        '''
        '''
        if there are more than 2 nodes in a cluster
        '''
        if len(Gs) > 1:            
#            '''
#            find subgraphs using edge distance match
#            '''
#            GMl = iso.GraphMatcher(Gl, Gs, edge_match=iso.numerical_edge_match(['length'],[1.0]))
#            '''
#            find subgraphs using node layer match
#            '''
#            GMz= iso.GraphMatcher(Gl, Gs, edge_match= iso.categorical_edge_match(['z'],[1.0])  )
#            '''
#            list down total number of subgraphs niso GMz||GMl
#            '''
#            x = [y for y in GMz.subgraph_isomorphisms_iter() if y in GMl.subgraph_isomorphisms_iter()]
            
            GMn = iso.GraphMatcher(Gl, Gs, node_match= iso.categorical_edge_match(['z'],[1]), 
                               edge_match= iso.numerical_edge_match(['length'],[1.0]))
            x = [y for y in GMn.subgraph_isomorphisms_iter()]
            
        else:
            '''
            find subgraphs using node layer match
            '''
            GMn = iso.GraphMatcher(Gl, Gs, node_match= iso.categorical_edge_match(['z'],[1]) )
            x = [y for y in GMn.subgraph_isomorphisms_iter()]
            
            
        niso =len(x)
        '''
        save subgraphs to a list
        '''
        subg = list()
        for i in range(niso):    
            subg.append(tuple(x[i].keys()))
        
        '''
        save product into a list 
        and caclulate the sum divid by total number of subgraphs
        '''
        subi = []
        subs = []
        for i in range(niso):
            subi.append([])
            for j in range(len(subg[i])):
                subi[i].append(self.get_occupancy(Gl,subg[i][j]))   
            subs.append(np.product(subi[i]))
        delta = np.sum(subs)/niso
        
        
        return delta, niso
    
    def get_delta_l(self, Gl, Gs):
    
        '''
        takes in larger graph Gl and smaller graph Gs
        find sub isomorphic graphs of Gs from Gl
        calculate the delta value in pi matrix 
        '''
        '''
        if there are more than 2 nodes in a cluster
        '''
        niso =len(Gs)
        ncluster = len(Gs[0]) #size of the clusters
        '''
        save product into a list 
        and caclulate the sum divid by total number of subgraphs
        '''
        subi = []
        subs = []
        for i in range(niso):
            subi.append([])
            for j in range(len(Gs[i])):
                subi[i].append(self.get_occupancy(Gl,Gs[i][j]))   
            subs.append(np.product(subi[i]))
        delta = np.sum(subs)
    
        return delta, niso
    
    def get_pi_matrix_G(self, G1v, G2v):
        '''
        The function that gets 
            
            configuration graphs, G1v
            cluster graphs, G2v
        and returns the interaction correlation matrix pi
        '''
        n1 = len(G1v)
        n2 = len(G2v)
        pi = np.zeros((n1,n2))
        niso_m = np.zeros((n1,n2))
        progress = 0
        
        for i in range(n1):
            for j in range(n2):
                pi[i][j], niso_m[i][j] = self.get_delta_G(G1v[i],G2v[j])
                
                progress = progress + 1
                per = progress/n1/n2 *100
                #print('%.2f %% done!' %per)
                        
        self.pi = pi
        self.niso_m = niso_m
        
        return pi
    
    def get_pi_matrix_l(self, G1v, G2v, print_progress = False):
        '''
        The function that gets 
            
            configuration graphs, G1v
            cluster graphs, G2v
        and returns the interaction correlation matrix pi
        '''
        n1 = len(G1v)
        n2 = len(G2v)
        pi = np.zeros((n1,n2))
        niso_m = np.zeros((n1,n2))
        progress = 0
        
        for i in range(n1):
            for j in range(n2):
                pi[i][j], niso_m[i][j] = self.get_delta_l(G1v[i],G2v[j])
                
                if print_progress:
                    progress = progress + 1
                    per = progress/n1/n2 *100
                    print('%.2f %% done!' %per)
                        
        self.pi = pi
        self.niso_m = niso_m
        
        return pi
    
    def get_J(self, Ev):
        '''
        The function input energy of configurations, Ev
        Returns cluster energy J from linear regression
        '''
        self.Ev = np.array(Ev)
        J = np.linalg.lstsq(self.pi, self.Ev)[0]
        self.J = J
        
        return J
    
    def get_MSE(self):
        '''
        Returns MSE of prediction and real cluster energy
        '''
        ns = len(self.Ev)    
        MSE = np.sum(np.power((np.dot(self.pi,self.J) - self.Ev),2))/ns
        
        self.MSE = MSE
        
        return MSE

#%%
class subgraphs():
    '''
    generate subgraph list with the nodes numbers under the mother graph
    '''
    
    def __init__(self, mother, dz):
        
       self.index= np.arange(len(mother)) # generate the index of nodes
       self.mother = mother
       self.dz = dz
       
    @staticmethod
    def layer_tuple(mother, dz, ci):
        
        '''
        takes in a combo of index and returns tuple of layers they are in 
        '''
    
        n = len(ci)
        index = []
        for i in range(n):
            index.append(ci[i])
            
        layers = []
        
        for i in range(n):
            layers.append(int(mother[index[i]][2]/dz))
            
        layers= tuple(layers)
            
        return layers
    
    @staticmethod   
    def distance_tuple(mother, ci):
        '''
        takes in a combo of index and returns sorted distance between nodes
        '''
        n = len(ci)
        index = []
        
        for i in range(n):
            index.append(ci[i])
        
        combo = list(combinations(index,2))
        ncombo = len(combo) #0 for 1 node, 2 for 2 nodes, 3 for 3 nodes
          
        distances = []
        
        for i in range(ncombo):
            pt1 = mother[combo[i][0]]
            pt2 = mother[combo[i][1]]
            distances.append(two_points_D(pt1, pt2))
            
        distances = tuple(sorted(distances))
        
        return distances
    
    @staticmethod
    def unique_combo(combo, indices_list):
    
        Gv_list = []
        nclusters = len(indices_list)
        
        for i in range(nclusters):
            Gv_list.append([])
            niso = len(indices_list[i])
            for j in range(niso):
                Gv_list[i].append(combo[indices_list[i][j]])
        
        return Gv_list


    def get_s(self, n_atoms):
        
        '''
        Input number of nodes in a subgraph
        Generate combinations among the nodes
        '''
        self.n_atoms = n_atoms
        
        combo = list(combinations(self.index, self.n_atoms))
        ncombo  = len(combo)
        
        
        '''
        generate the inform2tion list
        store the sorted distance of nodes in tuple 1
        + the layer each node is in in tuple 2
        '''
        
        info = [] 
        
        for i in range(ncombo):
            ci  = combo[i]
            
            distances = self.distance_tuple(self.mother, ci)
            layers = self.layer_tuple(self.mother, self.dz, ci)
            
            info.append((distances, layers))
        
        info_set = list(set(info))
        
        index_list =[]
        
        for i in info_set:
            index_list.append(info.index(i))
            
        index_list.sort() # sort the list and take out those indices
        
        s_np = np.array(combo)[index_list]
        
        '''
        convert 2D np array to list
        '''
        
        s_list = []
        for i in range(s_np.shape[0]):
            s_list.append(list(s_np[i]))
            
            
        return s_list
    
    def get_s2(self, n_atoms):
        
        '''
        Input number of nodes in a subgraph
        Generate combinations among the nodes
        '''
        print(n_atoms)
        self.n_atoms = n_atoms
        
        combo = list(combinations(self.index, self.n_atoms))
        ncombo  = len(combo)
        
        
        '''
        generate the information list
        store the sorted distance of nodes in tuple 1
        + the layer each node is in in tuple 2
        '''
        
        info = [] 
        
        
        for i in range(ncombo):
            
            ci  = combo[i]
            
            distances = self.distance_tuple(self.mother, ci)
            layers = self.layer_tuple(self.mother, self.dz, ci)
            
            info.append((distances, layers))
        
        info_set = list(set(info))
        #print(info_set)
        
        index_list =[]
        indices_list = []
        
        for i in info_set:
            index_list.append(info.index(i))
        
        index_list.sort() # sort the list and take out those indices
            
        for i in index_list:
            indices_list.append([a for a, x in enumerate(info) if x == info[i]])
        
        Gcv_list = self.unique_combo(combo, indices_list)    
           
        return Gcv_list
    
    def get_s3(self, n_atoms, cutoff_distance):
        
        '''
        Input number of nodes in a subgraph
        Generate combinations among the nodes
        '''
        self.n_atoms = n_atoms
        
        combo = list(combinations(self.index, self.n_atoms))
        ncombo  = len(combo)
        
        
        '''
        generate the information list
        store the sorted distance of nodes in tuple 1
        + the layer each node is in in tuple 2
        '''
        
        info = [] 
        combo_NN = []
        
        for i in range(ncombo):
            ci  = combo[i]         
            distances = self.distance_tuple(self.mother, ci)
            if np.max(np.array(distances)) > cutoff_distance:
                continue
            else:
                layers = self.layer_tuple(self.mother, self.dz, ci)
                info.append((distances, layers))
                combo_NN.append(ci)
                print('{}-body {} % done!'.format(n_atoms, np.round(i/ncombo*100, decimals = 3)))

        info_set = list(set(info))
        #print(info_set)
        
        index_list =[]
        indices_list = []
        
        for i in info_set:
            index_list.append(info.index(i))
        
        index_list.sort() # sort the list and take out those indices
            
        for i in index_list:
            indices_list.append([a for a, x in enumerate(info) if x == info[i]])
        
        Gcv_list = self.unique_combo(combo_NN, indices_list)    
           
        return Gcv_list
    
    
    def generate_clusters(self, cutoff_distance, up_to_nbodies = 3,  saveas_json = True):
        '''
        Generate clusters up to 3 body interactions 
        and save as a json file
        '''
        self.Gcv1 = self.get_s2(1)  # 1-body interaction
        self.Gcv2 = self.get_s3(2, cutoff_distance)  # 2-body interaction
        self.Gcv3 = self.get_s3(3, cutoff_distance)  # 3-body interaction
        self.Gcv = self.Gcv1 + self.Gcv2 + self.Gcv3
        
        self.count = dict()
        self.count['1 body'] = len(self.Gcv1)
        self.count['2 body'] = len(self.Gcv2)
        self.count['3 body'] = len(self.Gcv3)
        
        if up_to_nbodies == 4:
            self.Gcv = self.Gcv + self.get_s2(4)
            
        if saveas_json:
            
            #convert to jsonable format
            Gcv_jsonable = []
            for Gcv_i in self.Gcv:
                Gcv_i_jsonable = []
                for Gcv_j in Gcv_i:
                    Gcv_j_jsonable = [int(g) for g in Gcv_j]
                    Gcv_i_jsonable.append(Gcv_j_jsonable)   
                    
                Gcv_jsonable.append(Gcv_i_jsonable)
            
            Gcv_dict = {'Gcv': Gcv_jsonable}
            with open('clusters.json', 'w') as outfile:
                    json.dump(Gcv_dict, outfile)
        

#%%
class coordination():

    '''
    calculate the coordination number (CN1, CN2) 
    and the general coordination number (GCN)
    Use for CO oxidation onto clusters graph, not in use anymore 
    '''
    
    def __init__(self,occupancy):
        
        '''
        takes in the occupancy color vector 
        occupancy[0] is the empty color
        occupancy[1] is the filled color
        '''
        
        self.occupancy = occupancy
        self.empty = self.occupancy[0]
        self.filled = self.occupancy[1]
    
    def num_1NN(self, G, i):
    
        '''
        G is a networkx graph
        i is the index of node in G
        
        returns number of nearest neighbors of node i 
        and a list of neighbor index number
        '''
        n_1NN = 0   
        list_1NN = []
        if G.nodes[i]['color'] == self.filled:  # check if the node is occupied 
            for j in list(G.neighbors(i)): #iterate through 1st NN
                if G.nodes[j]['color'] == self.filled: #check if the node is occupied
                    n_1NN = n_1NN + 1
                    list_1NN.append(j)
        else:
            print('No atoms detected at this position') 
        return n_1NN, list_1NN
    
    def num_2NN(self, G,i):
        
        '''
        G is a networkx graph
        i is the index of node in G where CO adsorbs
        
        returns a list of numbers of 2nd nearest neighbors of node i for each 1NN
        and a 2D list of 2NN index numbers
        '''
        n_2NN = []
        list_2NN = []
        if G.nodes[i]['color'] == self.filled: # check if the node is occupied 
            for j in G.neighbors(i):      # iterate through 1st NN
                if G.nodes[j]['color'] == self.filled: # check if the node is occupied 
                    n_2NN.append(self.num_1NN(G,j)[0]) # Add number of 2NN for 1NNs
                    list_2NN.append(self.num_1NN(G,j)[1]) # Add neighbor index number
        else:
            print('No atoms detected at this position')            
        return n_2NN, list_2NN
    
    def cal_CN1(self, G,COsites):
        
        '''
        G is a networkx graph
        COsites are the index of adsorption site
        
        returns 1st CN number
        '''
        
        CN1 = []
        sitetype = len(COsites)
        for i in range(sitetype):
            CN1.append(self.num_1NN(G, COsites[i])[0])
            
        '''
        take arithmaric mean for CN1 for bridge and hollow sites
        '''
        
        CN1 = np.mean(np.array(CN1))   
        
        return CN1
    
    
    def cal_CN2(self, G,COsites):
           
        '''
        G is a networkx graph
        COsites are the index of adsorption site
        
        returns 2nd CN number
        '''
        
        
        list_CN2 = []
        CN2 = []
        
        sitetype = len(COsites)
        
        for i in range(sitetype):
            list_CN2.append(self.num_2NN(G, COsites[i])[0])
            
        '''
        sum up 2NN numbers for each 1NN
        '''
        
        for i in range(sitetype):
            if len(list_CN2[i]) == 0: CN2.append(0)
            else: CN2.append(np.sum(np.array(list_CN2[i])))
        
        '''
        take arithmaric mean for CN2 for bridge and hollow sites
        '''
        
        CN2 = np.mean(np.array(CN2)) 
        
        return CN2
    
    def cal_GCN(self, G, COsites):
        
        '''
        G is a networkx graph
        COsites are the index of adsorption site
        
        returns general coordination number
        '''
        
        GCN = []
        
        sitetype = len(COsites)
        list_1NN = []
        '''
        find all avaiable 1NN index 
        '''
        for i in range(sitetype):
            list_1NN = list_1NN + self.num_1NN(G, COsites[i])[1]
        
        '''
        Use set to avoid double counting
        '''    
        list_1NN = list(set(list_1NN))
        '''
        Get CN for these 1NN nodes
        '''
        for i in list_1NN:
            GCN.append(self.num_1NN(G,i)[0])
    
        '''
        Set weight based on Pd(111)
        '''
        
        if len(COsites) == 1: weight = 12
        if len(COsites) == 2: weight = 18
        if len(COsites) == 3: weight = 22
        
        GCN = np.sum(np.array(GCN))/weight
    
        return GCN
    
    
    def num_Ce1NN(self, G, i):
        
        '''
        G is a networkx graph
        i is the index of node in G
        
        returns the flag of whether the atom is next to a Ce atom
        '''
        
        nCe = 0 
        if G.nodes[i]['color'] == self.filled: # check if the node is occupied
            if G.nodes[i]['z'] == '1':  # check if the node is in base layer
                nCe = 1 # 1 means the atom is in contact with 3 Ce atoms underneath
        return nCe
            
    def num_Ce2NN(self, G, i):
        
        '''
        G is a networkx graph
        i is the index of node in G
        
        returns the number of 2nd nearest Ce neighbors of node i 
        and a list of atom adjacent to Ce base layer
        '''
    
        n_2NN = 0   
        list_2NN = []
        if G.nodes[i]['color'] == self.filled: # check if the node is occupied
            for j in list(G.neighbors(i)): # iterate through 1st NN
                n_2NN = n_2NN + self.num_Ce1NN(G,j)  # check if the node is next to Ce
                if self.num_Ce1NN(G,j): list_2NN.append(j) # Append the index to a list
        else:
            print('No atoms detected at this position') 
        return n_2NN, list_2NN
    
    def cal_CeCN1(self, G, COsites):
        
        '''
        G is a networkx graph
        COsites are the index of adsorption site
        
        returns coordination number of Ce
        '''
        
        CN1 = []
        sitetype = len(COsites)
        
        for i in range(sitetype):
            CN1.append(self.num_Ce1NN(G, COsites[i]))
        
        '''
        take arithmaric mean for CN2 for bridge and hollow sites
        '''
        
        CN1 = np.mean(np.array(CN1)) * 3 # each Pd atom is coordinate by 3 Ce
        
        return CN1
        
    def cal_CeCN2(self, G, COsites):
        
        '''
        G is a networkx graph
        COsites are the index of adsorption site
        
        returns 2nd coordination number of Ce
        '''
        
        CN2 = []
        sitetype = len(COsites)
        
        for i in range(sitetype):
            CN2.append(self.num_Ce2NN(G, COsites[i])[0])
        
        '''
        take arithmaric mean for CN2 for bridge and hollow sites
        '''
        
        CN2 = np.mean(np.array(CN2)) *3 
           
        return CN2
    
    def cal_CeGCN(self, G, COsites):
    
        '''
        G is a networkx graph
        COsites are the index of adsorption site
        
        returns general coordination number of Ce
        '''
        
        GCN = []
        
        sitetype = len(COsites)
        list_1NN = []
        
        '''
        find all avaiable 1NN next to Ce index 
        '''
        
        for i in range(sitetype):
            list_1NN = list_1NN + self.num_Ce2NN(G, COsites[i])[1]
            
        list_1NN = list(set(list_1NN))
        
        '''
        Check if Ce is around for 1NN nodes
        '''
        
        for i in list_1NN:
            GCN.append(self.num_Ce1NN(G,i))
        
        
        if len(COsites) == 1: weight = 3
        if len(COsites) == 2: weight = 5
        if len(COsites) == 3: weight = 6
        
        GCN = np.sum(np.array(GCN))/weight * 3
    
        return GCN
    
    def get_CNs(self, G, COsites):
        
        '''
        Take in configuration G
        CO adsorption configuration index list 
        and CO sites index list
        add properties to the self object
        '''
        self.CN1 = self.cal_CN1(G,COsites)
        self.CN2 = self.cal_CN2(G,COsites)
        self.GCN = self.cal_GCN(G,COsites)

        self.CeCN1 = self.cal_CeCN1(G,COsites)
        self.CeCN2 = self.cal_CeCN2(G,COsites)
        self.CeGCN = self.cal_CeGCN(G,COsites)
   
    def get_z(self, G, COsites):
        
        '''
        Take in configuration G
        CO adsorption configuration index list 
        and CO sites index list
        add average layer number to the self object
        '''
        list_z = []
        sitetype = len(COsites)
        
        for i in range(sitetype):
            list_z.append(int(G.nodes[COsites[i]]['z']))
            
            
        self.z = np.mean(np.array(list_z))    
            
        
   
 
#%%    
class isomorphs():
    
    '''
    A class object to generate isomorphs at given configurations
    '''
    
    def __init__(self, mother, dz):
        '''
        takes in essential variables from structure constants
        '''
        self.mother = mother
        self.dz = dz


    def get_iso_config(self, config_list, Ec_list, i_config, saveas_json = False, drawing_flag = False):    
        '''
        Take each configuration (the configuration list, their energies and its index)
        check isomorphoric subgraphs
        and return a dictionary of nodes list
        '''
        config_i = config_list[i_config]
        n_nodes = len(config_i)
        n_layers = get_layers(self.mother, self.dz, config_i)
        node_layer_dict = get_node_layer_dict(self.mother, self.dz)
        node_index = []
        #print(node_layer_dict)
        for i in range(n_layers):
            node_index = node_index + node_layer_dict[i]
        
        sub_mother = self.mother[np.array(node_index)]
    
        
        Clusters = initialize_graph_object(sub_mother, self.dz)
        
        # Generate the mothe graph
        G1 =  Clusters.Gm
        # Generate the configuration graph
        Clusters.get_clusters(sub_mother, [config_i]) #one in layer 3 and one in layer 4
        Gcv = Clusters.Gcv
        G2 = Gcv[0]
        
        if drawing_flag == True:
            plt.figure()
            drawing(G1)
            
            plt.figure()
            drawing(G2)
        
        # Detect isomorphirc subgraphs 
        # the matching graphs are stored as key-value dictionary pairs
        if len(G2) > 1:
            GMn = iso.GraphMatcher(G1, G2, node_match= iso.categorical_edge_match(['z'],[1]), 
                                   edge_match= iso.numerical_edge_match(['length'],[1.0]))
            iso_matches = [y for y in GMn.subgraph_isomorphisms_iter()]
            #GMz= iso.GraphMatcher(G1, G2, edge_match= iso.categorical_edge_match(['z'],[1.0])  )
            #GMl = iso.GraphMatcher(G1, G2, edge_match= iso.numerical_edge_match(['length'],[1.0])  )
            #iso_matches = [y for y in GMz.subgraph_isomorphisms_iter() if y in GMl.subgraph_isomorphisms_iter()]
            
        else:
            GMn = iso.GraphMatcher(G1, G2, node_match= iso.categorical_edge_match(['z'],[1]) )
            iso_matches = [y for y in GMn.subgraph_isomorphisms_iter()]
        
        # Use set and np.unqiue to eliminate the repeated graphs
        # return the graphs in indices list
        
        iso_indices = [list(xi.keys()) for xi in iso_matches]
        iso_indices = [list(yi) for yi  in list(set(xi) for xi in iso_indices)]
        iso_indices = [list(xi) for xi in np.unique(iso_indices, axis = 0)]
        
        #Take all configurations with points fall on the right panel
        # as cluster expansion can handle sysmetric graphs
        #check if x of all the point > 0
        iso_indices_pos = []
        for iso_i in iso_indices:
            cond1 = np.any(self.mother[np.array(iso_i)][:,0] > 0)
            cond2 = np.all(self.mother[np.array(iso_i)][:,0] == 0)
            if cond1 or cond2:
                iso_indices_pos.append(sorted([int(xi) for xi in iso_i]))
    
        niso = len(iso_indices_pos)
        # save to a json file
        if saveas_json:
            output_dict = {'configuration': config_i,
                           'index': int(i_config),
                           'n_nodes': int(n_nodes),
                           'n_layers': int(n_layers),
                           'n_iso': int(niso),
                           'iso_graph_list': iso_indices_pos}
            
            with open('iso_config_' + str(i_config) +'.json', 'w') as outfile:
                json.dump(output_dict, outfile)
        else:
              E_iso_i = list(Ec_list[i_config] * np.ones(niso))
              return E_iso_i, iso_indices_pos
          
    def generate_all_iso(self, config, Ec, file_index = 0):
        '''
        Main part of the function
        Take in some configurations and their corresponding energies
        Generate all isomorphs
        '''
        E_iso = []
        config_iso = []
        for i in range(len(config)):
            E_iso_i, iso_indices_pos = self.get_iso_config(config, Ec, i, saveas_json = False, drawing_flag =  False) 
            print('{} batch {} % config done!'.format(file_index, i/len(config)*100))
            E_iso = E_iso + E_iso_i
            config_iso = config_iso + iso_indices_pos
            
        # Attach to self properties
        self.E_iso = E_iso
        self.config_iso = config_iso
        '''
        save to one json file, containing all iso config list and their energies
        '''
        
        file_name = 'ES_iso_' + str(file_index) + '.json'
        
        ES_dict = {'E_iso': E_iso, 'config_iso': config_iso}
        with open(file_name, 'w') as outfile:
                json.dump(ES_dict, outfile)

                
#%% 
class graphs_CO():
    '''
    Graph object for CO-CO interactions
    '''
    
    def __init__(self, occupancy, NN1, unit_length):
        
        '''
        takes in the occupancy color vector 
        occupancy[0] is the empty color
        occupancy[1] is the filled color
        unit length is the distance between Pd atop - Pd atop site
        '''
        self.unit_length = unit_length
        self.occupancy = occupancy
        self.empty = self.occupancy[0]
        self.filled = self.occupancy[1]
        self.NN1 = NN1
        #self.draw = draw
        
    def gmothers(self, mother, sitetype_list):
    
        '''
        takes in mother cooridate list 
        returns connected lattice graph
        '''
        #draw_mother = self.draw[0]
        self.mother = mother
        self.nm = len(mother)
        self.sitetype_list = sitetype_list
        Gm = nx.Graph()
        
        for i in range(self.nm):
            Gm.add_node(i, pos = self.mother[i], sitetype = self.sitetype_list[i], color = self.empty)
        
        
        self.edge = []
        self.edge_d = []
        self.edge_type = []
        
        # Save all egdes into a list and calculate the edge distance
        for i in range(self.nm):
            for j in np.arange(i+1,self.nm):
                self.edge.append((i,j))
                self.edge_d.append(two_points_D(mother[i],mother[j]))
                self.edge_type.append((self.sitetype_list[i], self.sitetype_list[j]))
                
                
        self.ne = len(self.edge)
        for i in range(self.ne): 
            
            if self.NN1: # only connect within 1st Nearest Neighbors 
                if self.edge_d[i] <= self.unit_length: 
                    Gm.add_edges_from([self.edge[i]], length = self.edge_d[i], edge_type = self.edge_type[i])

            
            else: # Add all edges, connect all nodes
                Gm.add_edges_from([self.edge[i]],  length = self.edge_d[i], edge_type = self.edge_type[i])
                #Gm.add_edges_from([self.edge[i]],  z = self.edge_z[i], length = self.edge_d[i], edge_type = self.edge_type[i])
        
#        if draw_mother:
#            drawing(Gm)
#            plt.title('%d lattice points' %self.nm)
        
        return Gm
    
    
    def gconfigurations(self, son):
        
        '''
        takes in mother coordinate list and son's index number and occupancy vector
        returns the shaded son graph
        '''   
        # draw_config = self.draw[1]
        ns = len(son)
        Gs = nx.Graph()

        for i in son:
            Gs.add_node(i, pos = self.mother[i], sitetype = self.sitetype_list[i], color = self.filled)

        cedge = []
        cedge_d = []
        cedge_type = []
            
        # Save all egdes into a list and calculate the edge distance
        for i in range(ns):
            for j in np.arange(i+1, ns):
                
                pt1_i = son[i] # the index of point 1
                pt2_i = son[j] # the index of point 2
                cedge.append((pt1_i, pt2_i))
                cedge_d.append(two_points_D(self.mother[pt1_i], self.mother[pt2_i]))
                cedge_type.append((self.sitetype_list[pt1_i], self.sitetype_list[pt2_i]))
              
        cne = len(cedge)
        
        for i in range(cne): 
            
            if self.NN1: # only draw 1st Nearest Neighbors 
                if cedge_d[i] <= self.unit_length:
                    Gs.add_edges_from([cedge[i]], length = cedge_d[i], edge_type = cedge_type[i])
            
            else: 
                Gs.add_edges_from([cedge[i]], length = cedge_d[i], edge_type = cedge_type[i])
#        # Can only draw 2D
#        if draw_config:
#            drawing(Gs)
#            plt.title('Pd %d' %ns)
        
        return Gs
            
            
            
            
            