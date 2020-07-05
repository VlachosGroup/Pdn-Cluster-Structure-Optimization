# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:38:48 2019

@author: wangyf
"""

'''
Binding Energy of CO Evaulation Functions for Pdm(CO)n clusters
'''


import os
import sys
import glob

import pickle
import json
import pandas as pd
import numpy as np
from sympy import Plane, Point3D
from itertools import combinations

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Slient the sklearn warnings
import warnings
warnings.filterwarnings('ignore') 

from ase import Atom
from ase.data import covalent_radii
from ase.io import read, write
from ase.visualize import view

import platform
HomePath = os.path.expanduser('~')
ProjectPath = os.path.join(HomePath, 'Documents', 'GitHub', 
                           'Pdn-Cluster-Structure-Optimization')

# Energy model directory
energy_path = os.path.join(ProjectPath, 'lasso-assisted-CE')
sys.path.append(energy_path)

import energy_functions as energy
from Pdbulk import NN1,NN2
import lattice_functions as lf



#%% I/O files


def save_CONTCAR(Pdi, index, atoms, output_dir):

    '''
    Save the configurations in POV
    '''

    filename = 'Pd'+str(Pdi) +'-' + str(index) + '-CONTCAR'

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    write(os.path.join(output_dir, filename), atoms)


def save_POV(Pdi, index, atoms, output_dir):

    '''
    Save the atom object
    '''
    pov_args = {
    	'transparent': True, #Makes background transparent. I don't think I've had luck with this option though
        'canvas_width': 900., #Size of the width. Height will automatically be calculated. This value greatly impacts POV-Ray processing times
        'display': False, #Whether you want to see the image rendering while POV-Ray is running. I've found it annoying
        'rotation': '0x, 0y, 90z', #Position of camera. If you want different angles, the format is 'ax, by, cz' where a, b, and c are angles in degrees
        'celllinewidth': 0.02, #Thickness of cell lines
        'show_unit_cell': 0 #Whether to show unit cell. 1 and 2 enable it (don't quite remember the difference)
        #You can also color atoms by using the color argument. It should be specified by an list of length N_atoms of tuples of length 3 (for R, B, G)
        #e.g. To color H atoms white and O atoms red in H2O, it'll be:
        #colors: [(0, 0, 0), (0, 0, 0), (1, 0, 0)]
        }
    #Write to POV-Ray file
    filename = 'Pd'+str(Pdi) +'_' + str(index) + '.POV'
    write(os.path.join(output_dir, filename), atoms, **pov_args)


def remove_CO(CONTCAR_filename, view_flag = True):
    '''
    Read the old CONTCAR file with a CO onto it
    remove the CO and save as a new CONTCAR
    '''
    old_name = CONTCAR_filename #'pd20-ceria-co-CONTCAR'
    atoms = read(old_name)
    

    # find number of Pd
    # find C atom index
    nPd = 0

    for i, atom in enumerate(atoms):
        if atom.symbol == 'Pd':
            nPd = nPd + 1
        if atom.symbol == 'C':
            C_in_CO = i

    C_O_Dist  = []
    O_in_CO  = []

    for k, atom in enumerate(atoms):
        if atom.symbol == 'O':
            dist = atoms.get_distance(C_in_CO, k)
            C_O_Dist.append(dist)
            O_in_CO.append(k)

    O_in_CO = O_in_CO[C_O_Dist.index(min(C_O_Dist))]

    del atoms[[O_in_CO, C_in_CO]]
    write('pd'+str(nPd)+'-no-CO-CONTCAR', atoms)
    
    # View the atom object if the flag is true
    if view_flag:
        view(atoms)



#%% Functions handling ase atoms objects

'''
Covalent bond lengths that might be useful
'''
Pdr = covalent_radii[46]
Or = covalent_radii[8]
Cr = covalent_radii[6]
CO = Cr + Or
PdO = Pdr + Or
PdC = Cr + Pdr - 0.5
unit_length = Pdr*2

def sort_i_and_d(D,I):
    '''
    Sort I based on sorted D indices
    '''
    Dsort = np.sort(D)
    Dsort = list(Dsort)
    D = list(D)
    Isort = []
    for d in Dsort:
        Isort.append(I[D.index(d)])

    return Dsort,Isort

def find_all_Pd(atoms):
    '''
    Count number of atoms, return all Pd atom indices
    '''
    Pd_indices = []

    for i, atom in enumerate(atoms):
        if atom.symbol == 'Pd': Pd_indices.append(i)

    return Pd_indices


def find_bridge_pairs(Pd_pairs, atoms):

    bridge_pairs = []
    for pair in Pd_pairs:
        Pd_Pd = atoms.get_distances([pair[0]], [pair[1]])
        if np.logical_and(Pd_Pd>=NN1[0], Pd_Pd<=NN1[1]):
            bridge_pairs.append(list(pair))

    return bridge_pairs

def find_hollow_triples(Pd_triples , atoms):

    hollow_triples = []
    for triple in Pd_triples:
        Pd_Pd1 = atoms.get_distances(triple[0], [triple[1], triple[2]])
        Pd_Pd2 = atoms.get_distances([triple[1]], [triple[2]])
        flag1 = np.logical_and(Pd_Pd1>=NN1[0], Pd_Pd1<=NN1[1])
        flag2 = np.logical_and(Pd_Pd2>=NN1[0], Pd_Pd2<=NN1[1])

        if np.all(list(flag1)+list(flag2)):
            hollow_triples.append(list(triple))

    return hollow_triples


def find_surface_Pd(atoms):
    '''
    Given an atoms object and detect all Pd atoms exposed at the surface
    return surface Pd atom indices in the atoms object
    '''

    atoms_obj = PdnCO()
    atoms_obj.atoms_descriptors(atoms)
    Pdpos = atoms_obj.Pdpos #Pd positions in atoms
    Pdi = atoms_obj.Pdi #Pd indices in atoms
    z_values = np.unique(np.around(Pdpos[: , 2], decimals = 0)) #unique z values
    zi = [] #layer index starting from 0
    NN1_v = [] #number of NN1 for Pd atom
    Pd_surface = []

    for i, atomi in enumerate(Pdi):
        zi.append(np.where(np.abs(Pdpos[i, 2] - z_values) < 0.5)[0][0])
        NN1_v.append(atoms_obj.PdNN.loc['NN1']['Pd'+str(atomi)])

    for i, atomi in enumerate(Pdi):

        if zi[i] == 0 and NN1_v[i] == 9: continue #9 is the max coordination for base layer Pd
        if zi[i] > 0 and NN1_v[i] == 12: continue #12 is the max for higher level Pd

        Pd_surface.append(atomi)

    return Pd_surface



def find_sites(Pd_interest, atoms):
    '''
    Input incides for interested Pd atoms
    and the atoms object
    '''
    
    #Find all top CO adsorption sites
    top_sites = []
    for Pdi in Pd_interest: top_sites.append([Pdi])
    CO_sites_list = top_sites

    #Find all bridge and hollow
    
    bridge_sites = []
    Pd_pairs  = list(combinations(Pd_interest,2))
    bridge_sites = find_bridge_pairs(Pd_pairs, atoms)


    hollow_sites = []
    Pd_triples  = list(combinations(Pd_interest,3))
    hollow_triples = find_hollow_triples(Pd_triples, atoms)
    hollow_sites =   hollow_triples

    CO_sites_list = CO_sites_list + bridge_sites + hollow_sites

    return CO_sites_list

def find_all_surface_sites(atoms):
    '''
    Input atoms object
    return all surface sites
    '''

    Pd_surface = find_surface_Pd(atoms)
    COsites_list = find_sites(Pd_surface, atoms)

    return COsites_list

def find_all_top_sites(atoms):
    '''
    Input atoms object
    return all top sites
    '''
    Pd_top = find_all_Pd(atoms)
    COsites_list =  [[Pdi] for Pdi in Pd_top]

    return COsites_list


#%% Class for Pdn(CO) objects - only support one CO per cluster

class PdnCO():

    def __init__(self, CONTCAR_filename = []):

        '''
        Initializing descriptor variables
        '''
        # CONTCAR filename, [] if not CONTCAR file is input
        self.filename = CONTCAR_filename

        self.Eads = []
        self.charge = []
        self.realsite = []


    def io_descriptors(self, data):
        '''
        Input descriptor data from
        a data frame containing the additional electronic information
        Parse the input data
        '''

        self.Eads = float(data[data['Filename'] == self.filename]['Eads'])
        self.charge = float(data[data['Filename'] == self.filename]['Charge'])
        self.realsite = data[data['Filename'] == self.filename]['RealSite'].values[0]

    def atoms_descriptors(self, atoms):

        '''
        Takes in an atoms object
        Count number of atoms in the atom object
        '''
        self.atoms = atoms

        # Atom index
        Pdi = []
        Cei = []
        Ci = []
        Oi = []
        Pd_C = []

        for i, atom in enumerate(self.atoms):
            if atom.symbol == 'Pd': Pdi.append(i)
            if atom.symbol == 'Ce': Cei.append(i)
            if atom.symbol == 'C':  Ci.append(i)
            if atom.symbol == 'O':  Oi.append(i)


        #No of Pd atoms in the cluster
        self.NPd = int(len(Pdi))


        # Take out the O in CO, only consider lattice O
        if not Ci == []:
            C_O = self.atoms.get_distances(Ci[0], Oi, mic = True)
            Oi.pop(int(np.where(C_O == C_O.min())[0]))

            # all Pd-C bond length
            Pd_C = self.atoms.get_distances(Ci[0], Pdi, mic = True)
            # sorted Pd-C bond length
            Pd_C, Pdi = sort_i_and_d(Pd_C, Pdi)


        self.Pd_C = Pd_C
        self.Pdi = Pdi
        self.Cei = Cei
        self.Ci = Ci
        self.Oi = Oi



        '''
        Save number of NNs for Pd, Ce, O in pandas dataframe
        '''
        Pd_Pd = pd.DataFrame() #Pd to Pd bond length table for all Pd atoms
        PdNN = pd.DataFrame() #Pd NN table contain the number of 1NN and 2NN for all Pd atoms

        Pd1NN= dict() #Pd first NN table for Pd atoms at the site
        PdONN = dict()
        PdCeNN = dict()

        Pdpos = [] #Pd atom position

        # Iterate through each Pd atom on the lattice
        for i in self.Pdi:

           # Save Pd atom position
           Pdpos.append(atoms[i].position)

           # Find Pd Pd CNs
           Pd_Pd_D =  self.atoms.get_distances(i, self.Pdi)
           Pd_Pd_D, Pdisort = sort_i_and_d(Pd_Pd_D, self.Pdi)

           Pd_Pd['distance_from_'+str(i)] =  Pd_Pd_D
           Pd_Pd['i'+str(i)] = Pdisort
           PdNN['Pd'+str(i)] = [sum(np.logical_and(Pd_Pd_D>=NN1[0], Pd_Pd_D<=NN1[1])),
                sum(np.logical_and(Pd_Pd_D>=NN2[0], Pd_Pd_D<=NN2[1]))]
           Pd1NN['Pd'+str(i)] = np.array(Pdisort)[np.where(np.logical_and(Pd_Pd_D>=NN1[0],Pd_Pd_D<=NN1[1]))[0]]

           '''
           # Find the OCN for Pd atoms at the sites by setting fixed cut-off distance for PdO NN
           PdOD = self.atoms.get_distances(i, self.Oi)
           PdOD, _ = sort_i_and_d(PdOD, self.Oi)
           PdnO = len(np.where(np.array(PdOD) < 3.5)[0])
           PdONN['Pd'+str(i)] = PdnO

           # Find the CeNN for Pd atoms at the sites by setting fixed cut-off distance for PdCe NN
           PdCeD = self.atoms.get_distances(i, self.Cei)
           PdCeD, _ = sort_i_and_d(PdCeD, self.Cei)
           PdnCe = len(np.where(np.array(PdCeD) < 4.2)[0])
           PdCeNN['Pd'+str(i)] = PdnCe
           '''

        # Rename PdNN table
        PdNN.index = ['NN1','NN2']

        # Append to self object
        self.PdNN = PdNN
        self.Pd1NN = Pd1NN
        '''
        self.PdONN = PdONN
        self.PdCeNN = PdCeNN
        '''

        self.Pdpos = np.array(Pdpos)

    def get_COsites(self):

        '''
        Determine CO site if unknown
        and the indices of Pd atoms by real Pd-C bond length comparison
        '''
        Pd_C = self.Pd_C
        Pdi = self.Pdi
        #The distance of Pd to the first nearest C
        PdC3 = np.zeros(3)
        #The bond tolerance is
        bond_tol = 0.7
        if len(Pd_C) == 1:
            PdC3[0] = Pd_C[0]
            COsites = np.array(Pdi)[:1] #top

        if len(Pd_C) == 2:
            PdC3[:2] = Pd_C[:2]
            diff = Pd_C[1] - Pd_C[0]
            if diff < bond_tol: COsites = np.array(Pdi)[:2] #bridge
            else: COsites = np.array(Pdi)[:1] #top

        if len(Pd_C) >= 3:
            PdC3 = Pd_C[:3]
            diff1 = Pd_C[1] - Pd_C[0]
            diff2 = Pd_C[2] - Pd_C[1]
            if diff1 > bond_tol: COsites = np.array(Pdi)[:1] #top
            else:
                if diff2 > bond_tol: COsites = np.array(Pdi)[:2] #bridge
                else: COsites = np.array(Pdi)[:3] #hollow

        '''
        Obtain site specific information
        '''
        self.COsites = COsites

        # Bond lengths
        self.PdC1 = PdC3[0]
        self.PdC2 = PdC3[1]
        self.PdC3 = PdC3[2]

    def site_descriptors(self, COsites = []):
        '''
        Calculate site specific descriptors
        '''

        # when the site indices are not provided
        if COsites == []:
            self.get_COsites()

        else:
            self.COsites = COsites


        # Numbef of sites
        self.Nsites = len(self.COsites)
        if self.Nsites== 3: self.sitetype = 'hollow'
        if self.Nsites == 2: self.sitetype = 'bridge'
        if self.Nsites == 1: self.sitetype = 'top'

        #indices of Pd atom at CO binding sites
        COsites_Pdi = []
        for s in range(len(self.COsites)):
            COsites_Pdi.append('Pd'+str(self.COsites[s]))

        # Get CO site position - the mean of Pd pos at the site
        COsites_Pdpos = []
        for i in self.COsites: COsites_Pdpos.append(self.atoms[i].position)
        self.site_pos = np.mean(COsites_Pdpos, axis = 0)

        # Add a facticious C to the end
        atoms_C = self.atoms.copy()
        atoms_C.append(Atom('C', position = self.site_pos))
        # all Pd-site bond length
        Pd_site = atoms_C.get_distances(-1, self.Pdi, mic = True)
        # sorted Pd-site bond length
        Pd_site, _ = sort_i_and_d(Pd_site, self.Pdi)
        # Site distance to neighboring Pd atoms
        Pd_site_CO = np.array(Pd_site[:len(self.COsites)])

        #NN dataframe at CO binding site only
        PdNN_CO = self.PdNN.loc[:, COsites_Pdi]


        # Aprroximate Bond lengths by site-Pd length
        if not COsites == []:

            PdC3 = np.zeros(3)
            PdC3[:len(self.COsites)] = Pd_site_CO
            # Bond lengths
            self.PdC1 = PdC3[0]
            self.PdC2 = PdC3[1]
            self.PdC3 = PdC3[2]

        '''
        Weighted average for NN1, NN2, GCN
        '''
        #weights based on 1 over CO-Pd distance
        if self.Nsites == 1: # for top site
            norm_weights = np.ones(1)  #avoid zero division problem
        else:
            norm_weights = (1/Pd_site_CO)/np.sum(1/Pd_site_CO)

        # CN1 and CN2
        self.CN1 = np.dot(norm_weights, PdNN_CO.loc['NN1'].values)
        self.CN2 = np.dot(norm_weights, PdNN_CO.loc['NN2'].values)

        # GCNs
        cn_max = [12, 18, 22]
        NN1_site = []
        #Iterate through each atom at the site
        for i in self.COsites:
            for j in self.Pd1NN['Pd'+str(i)]:
                NN1_site += list(self.Pd1NN['Pd'+str(j)])
        #Find non-repeating NN1 atoms for the site
        NN1_site = list(set(NN1_site))
        # Take out the atoms at the site
        NN1_site = [ni for ni in NN1_site if ni not in list(self.COsites)]
        #Add up CN numbers for those NN1 atoms
        gcn_sum = 0
        for ni in NN1_site:
            gcn_sum += self.PdNN.loc['NN1']['Pd'+str(ni)]
        #Normalize by the max GCNs
        self.GCN = gcn_sum/cn_max[self.Nsites -1]

        '''
        '''
        #Weighted average for OCN and CeCN
        '''
        PdONN_CO = []
        PdCeNN_CO = []
        for si in COsites_Pdi:
            PdONN_CO.append(self.PdONN[si])
            PdCeNN_CO.append(self.PdCeNN[si])

        self.OCN1  = np.dot(np.array(PdONN_CO), norm_weights)
        self.CeCN1 = np.dot(np.array(PdCeNN_CO), norm_weights)
        '''

        '''
        Calculate distance to the support
        '''
        #take the  distance of CO to Ce plane (determined by 3 Ce points)
        # as the distance to support
        Ce_plane = Plane(Point3D(self.atoms[self.Cei[0]].position),
                         Point3D(self.atoms[self.Cei[1]].position),
                         Point3D(self.atoms[self.Cei[2]].position))
        self.Dsupport = float(Ce_plane.distance(Point3D(self.site_pos)))

    def gather_descriptors(self, atoms, COsites = [], data = None):
        '''
        Gather descriptors
        Make a row in dataframe as an ID for each structure including filenames and properties etc
        '''
        # input io data
        if not data == None:
            self.io_descriptors(data)

        # input atoms objects
        self.atoms_descriptors(atoms)

        # input COsites
        self.site_descriptors(COsites)



        self.structureID =  [self.filename, #filename
                             self.atoms, # atoms object
                             self.Eads, #Eads
                             self.NPd, #Npd
                             self.realsite, #real sitetype
                             self.sitetype, #sitetype from calculation
                             self.CN1, #CN1
                             self.CN2, #CN2
                             self.GCN, # general cooridination number
                             self.Dsupport, #Z
                             self.charge, #Bader charge
                             self.Nsites, #number of sites
                             self.PdC1, #1st Pd-C distance
                             self.PdC2, #2nd Pd-C distance
                             self.PdC3] #3rd Pd-C distance
                             #self.CeCN1,
                             #self.OCN1]


#%% Class object for PCA binding energy prediction given sites for a Pdn(CO)m cluster
# multiple COs supported
class be_regression_model():

    def __init__(self, model_name = 'spca'):

        self.model_name = model_name

        # import spca model
        if self.model_name == 'spca':
            estimator_file  = os.path.join(model_path, 'spca_estimator.p')
            [self.eig_vecs, self.scaler, self.estimator] = pickle.load(open(estimator_file,'rb'))

        # import pca model
        if self.model_name == 'pca':
            estimator_file  = os.path.join(model_path, 'pca_estimator.p')
            [self.pca, self.scaler, self.estimator]  = pickle.load(open(estimator_file,'rb'))

        # import random forest model 
        if self.model_name == 'rf':
            estimator_file = os.path.join(model_path, 'rf_estimator.p')
            [self.estimator, self.scaler] = pickle.load(open(estimator_file,'rb'))

        self.estimator_file = estimator_file

        # all correspond to the structureID in PdnCO
        self.structureID_labels = ['Filename', 'AtomsObject', 'Eads', 'NPd', 'RealSite',  'SiteType', 'CN1', 'CN2', 'GCN',
                                   'Z', 'Charge', 'Nsites', 'Pd1C', 'Pd2C', 'Pd3C'] #, 'CeCN1', 'OCN1']
        self.descriptors  =  ['NPd', 'CN1', 'CN2','GCN', 'Z', 'Nsites'] #,  'CeCN1', 'OCN1'] #8 geometric descriptors

    def cal_descriptor_data(self, atoms, COsites, save_csv =  False):

        de_data = pd.DataFrame(columns = self.structureID_labels)
        for i, site in enumerate(COsites):

            atoms_obj = PdnCO()
            atoms_obj.gather_descriptors(atoms, site)
            de_data.loc[i,:] = atoms_obj.structureID

        # Assign to self
        self.de_data = de_data

        # Output as a csv file
        if save_csv:
            de_data.to_csv('new_descriptor_data.csv', index=False, index_label=False)

    def predict_binding_E(self, atoms, COsites):
        '''
        The main function in be_model class to predict the binding energy
        ietratively call PdCO class by passing the atoms object and each CO site 
        '''
        # Calculate descriptor data first
        self.cal_descriptor_data(atoms, COsites)

        # Extract site types
        self.sitetype_list = list(self.de_data.loc[:,'SiteType'])


        # Find X for regression based on model name
        if self.model_name == 'spca':
            # Extract data into a matrix form
            self.X = np.array(self.de_data.loc[:, self.descriptors], dtype = float)
            # Standardize the data
            self.X_std = self.scaler.transform(self.X)
            # Tranform in spca
            self.Xreg = np.linalg.lstsq(self.eig_vecs.T, self.X_std.T, rcond=None)[0].T

        if self.model_name == 'pca':
            # Extract data into a matrix form
            self.X = np.array(self.de_data.loc[:, self.descriptors], dtype = float)
            # Standardize the data
            self.X_std = self.scaler.transform(self.X)
            # Use 6 pcs
            self.Xreg = self.pca.transform(self.X_std)[:,:6]

        if self.model_name == 'rf':
            # use only 5 descriptors 
            descriptors_rf = ['NPd', 'Nsites', 'Z',  'CN1', 'CN2'] #['NPd',  'CN1', 'CN2',  'Z', 'Nsites' ]
            self.X = np.array(self.de_data.loc[:, descriptors_rf], dtype = float)
            self.X_std = self.scaler.transform(self.X)
            self.Xreg = self.X_std.copy()

        # Predict y
        self.y = self.estimator.predict(self.Xreg)

        # Filter for negative binding energies
        self.y_bind = self.y.copy()
        self.y_bind[np.where(self.y>0)] = 0

        # Gather GCN (can be other descriptors)
        self.GCNs = np.array(self.de_data['GCN'], dtype = float)
        self.CN1 = np.array(self.de_data['CN1'], dtype = float)
        self.CN2 = np.array(self.de_data['CN2'], dtype = float)



class PdnCOm():
    '''
    Pdn cluster with mutiple CO object, input atoms is an atom object with bare Pd clusters
    '''
    def __init__(self, atoms, Pd_interest = [], top_only = False):

        self.atoms = atoms
        if Pd_interest == []:
            # Consider all the Pd atoms
            if top_only:
                self.COsites = find_all_top_sites(self.atoms)
            # Consider all the surface sites
            else:
                self.COsites = find_all_surface_sites(self.atoms)
        # find sites among specific Pd atoms         
        else:
            self.COsites = find_sites(Pd_interest, atoms)

        '''
        Get Pd indices and positions
        '''

        atoms_obj = PdnCO()
        atoms_obj.atoms_descriptors(atoms)
        self.Pdpos = atoms_obj.Pdpos #Pd positions in atoms
        self.Pdi = atoms_obj.Pdi #Pd indices in atoms

        # Get surface atom properties 
        self.Pd_surface = find_surface_Pd(self.atoms)  # Pd atoms on the surface
        self.ratio_surface = len(self.Pd_surface)/len(find_all_Pd(self.atoms)) # the ratio of surface Pd atoms to the total number of atoms


    def binding_Es(self, be_model):

        be_model.predict_binding_E(self.atoms, self.COsites)
        self.y_bind_list = be_model.y_bind
        self.sitetype_list = be_model.sitetype_list
        self.GCNs = be_model.GCNs
        self.CN1s = be_model.CN1
        self.CN2s = be_model.CN2

    def append_COs(self, ind_index, view_flag = False):

        '''
        Append the COs onto the clean Pd cluster atoms object
        '''

        # if no COs present
        if len(ind_index) == 0:
            atoms = self.atoms.copy()
            return atoms

        else:

            self.COsites_occ = np.array(self.COsites)[ind_index]
            self.nCO = len(self.COsites_occ)

            # Get CO site position - the mean of Pd pos at the site
            COsites_pos = []

            for sites_i in self.COsites_occ:  # iterate through all sites

                Pdpos = [self.atoms[j].position for j in sites_i]
                COsites_pos.append(np.mean(Pdpos, axis = 0))

            # Add PdC colvanet length to site pos to get C position
            CO_pos =  np.array(COsites_pos)
            CO_pos[:,2] = CO_pos[:, 2] + PdC

            # At each occupied site, append C to it
            atoms = self.atoms.copy()
            for i in range(self.nCO):
                atoms.append(Atom('C', position = CO_pos[i] ) )

            # assign CO position to atoms
            self.CO_pos = CO_pos

            if view_flag: view(atoms)

            # return the atoms object with multiple COs on it
            return atoms


def predict_binding_Es_fast(atoms, Pd_interest = [], ind_index = None, view_flag =  False, output_descriptor = False, top_only = False):

    '''
    Fast function to predict binding energy from spca
    return binding energy in a list and sitetype list
    '''

    PdnCOm_obj = PdnCOm(atoms, Pd_interest, top_only) # create PdnCOm object
    be_model = be_regression_model('rf')  # create the binding energy PCA model
    PdnCOm_obj.binding_Es(be_model) # combine the two
    binding_Es = PdnCOm_obj.y_bind_list # Output binding energy in a list
    COsites = PdnCOm_obj.COsites # Output CO sites in a list
    sitetype_list = PdnCOm_obj.sitetype_list # Output corresponding site type list
    GCNs = PdnCOm_obj.GCNs # Output corresponding GCN values in a list 
    CN1s = PdnCOm_obj.CN1s # Output corresponding CN1 values in a list 
    CN2s = PdnCOm_obj.CN2s # Output corresponding CN2 values in a list 
    ratio_surface = PdnCOm_obj.ratio_surface #Output corr


    # Append CO to the atoms object and append their positions
    CO_pos  = []
    if ind_index == None: ind_index = range(0, len(COsites))
    PdnCOm_obj.append_COs(ind_index, view_flag)
    CO_pos = PdnCOm_obj.CO_pos

    if output_descriptor:
        return binding_Es, COsites, CO_pos, sitetype_list, GCNs, CN1s, CN2s, ratio_surface
    else: 
        return binding_Es, COsites, CO_pos, sitetype_list


def check_CO_CO_distance(co_config, coi, CO_pos):
    '''
    input the indices of occupied CO sites and new proposed site
    check if the new CO is in the cut-off range next to the previous COs
    If the CO-CO interactions are provided, we dont need this function anymore
    '''
    
    acceptance_flag = True
    if len(co_config) < 1: # if no CO, no need to check!
            pass
    
    else:

        pt1 = CO_pos[coi]
        distance  = []
        for coj in co_config:
            if not coj == coi:
                distance.append(lf.two_points_D_np(pt1, CO_pos[coj]) )
        #try: 
        min_distance = np.min(distance)
        #except: min_distance = 0
    
        if min_distance <= unit_length/2: #set no tolerance
            acceptance_flag = False
        else: pass

    return acceptance_flag


def check_Pd_CO_distance(pd_chosen_empty_i, pd_chosen_occ_i, mother_with_support, co_config, co_pos):
    '''
    input the indices of proposed new Pd sites and Pd positions
    check if the Pd atom is close to any COs
    pd_chosen_empty_i - new Pd position
    pd_chosen_occ_i - old Pd position
    
    '''
    acceptance_flag = True

    # check if the old Pd is not attached to a CO
    pd_old = mother_with_support[pd_chosen_occ_i]
    pd_old_co_distance = [lf.two_points_D_np(pd_old, co_pos[i]) for i in co_config]

    if np.min(pd_old_co_distance) <= (PdC + 0.8): # hollow, bridge has longer PdC bonds
        acceptance_flag = False

        # check if the new Pd is not overlapping with a CO
        pd_new = mother_with_support[pd_chosen_empty_i]
        pd_new_co_distance = [lf.two_points_D_np(pd_new, co_pos[i]) for i in co_config]
        if np.min(pd_new_co_distance) < PdC  - 0.1: # set certain tolerance
            acceptance_flag = False
        else: pass
    else: pass


    return acceptance_flag

def update_COsites(Pdn_atoms_new, co_config_old, co_pos_old):
    '''
    Takes in the old and new Pdn atoms object
    update keep the COs at the original position and return the new CO one-hot index
    '''
    binding_Es, cosites, co_pos_new, sitetype_list_new = predict_binding_Es_fast(Pdn_atoms_new, view_flag = False)

    # When we have no CO to start with
    if len(co_config_old) == 0 :
        cox = np.zeros(len(cosites))
        co_config_new = co_config_old.copy()
    # When some sites are occupied by CO
    else:
        co_config_new = []
        co_occ_pos_old = co_pos_old[co_config_old]
        # Check the old occupied sites and new sites to see if they have overlap
        # Keep the indices of overlapping new sites as the new co_config
        for co_pos_i in co_occ_pos_old:
           for j, co_pos_j in enumerate(co_pos_new):
               # if the coordinates differ with in the tolerance
               if lf.two_points_D_np(co_pos_i, co_pos_j) <= 1e-5: co_config_new.append(j)
        # check if there is CO to be conserved
        if len(co_config_new) > 0:
            cox = energy.index_to_one_hot(co_config_new, len(cosites))
        else: 
            cox = np.array([])
            
    return cox, co_config_new,  binding_Es, cosites, co_pos_new, sitetype_list_new


