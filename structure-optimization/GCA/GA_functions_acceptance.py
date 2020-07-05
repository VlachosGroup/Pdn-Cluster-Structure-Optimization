# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:14:01 2018

@author: wangyf
"""

import random
import json

try:
    from mpi4py import MPI
except:
    pass

from datetime import datetime
from sklearn.metrics import mean_squared_error
from numpy.linalg import norm
from operator import attrgetter
from ase import Atoms

import os
import sys
import pandas as pd
import numpy as np
import pickle

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
from test_connectivity_fitness import structure_ids

n_nodes = len(super_mother)

# Define the constants

# Boltzmann constant
kb = 8.617333262145e-05

#%%

def import_deap(nfitness):
    '''
    import deap object and create individuals
    '''
    from deap import tools
    from deap import base
    from deap import creator
    score_weights = tuple(-1 * np.ones(nfitness)) #tuple for min-1.0, max+1.0
    #Create the fitness object
    creator.create("FitnessMin", base.Fitness, weights = score_weights)
    #Create an individual
    creator.create("Individual", list, fitness = creator.FitnessMin)

    return tools, base, creator


def get_rank(COMM = None):
    if COMM is None:
        return 0
    else:
        return COMM.rank

def get_size(COMM = None):
    if COMM is None:
        return 1
    else:
        return COMM.size

def get_time():
    return datetime.now()


def save_population(COMM = None, population = None, file_name = None):
	if COMM is None:
		rank = 0
	else:
		rank = COMM.rank

	if rank == 0:
		with open(file_name, 'wb') as f_ptr:
			pickle.dump(population, f_ptr)

def one_hot_to_index(individual):
    '''
    Convert an individual from one hot encoding to a list index
    '''
    ind_index = list(np.nonzero(individual)[0])
    return ind_index

def index_to_one_hot(ind_index, n_nodes = n_nodes):
    '''
    Convert an individual from a list index to one hot encoding
    '''
    individual = np.zeros(n_nodes, dtype = int)
    individual[np.array(ind_index)] = 1
    individual = list(individual)

    return individual


def config_pool(config_index_list):
    '''
    Generate the pool of configurations
    included in one hot encoding format
    '''
    config_one_hot_list = [index_to_one_hot(x[0]) for x in config_index_list]

    return config_one_hot_list

def novelty_score(individual, config_index_list):
    '''
    This function needed to be updated to consider the rotational and translational structures
    '''
    config_one_hot_list = config_pool(config_index_list)
    n_score = - np.min(np.linalg.norm(np.array(individual) - np.array(config_one_hot_list), axis =1))

    return n_score

def initial_individual(n_seeds, mother = super_mother, dz = dz):
    '''
    Takes in nseeds and create an individual
    with n_seeds occupied in the base layer
    '''
    # the index for base layer atoms in super cell
    base_indices = np.where(mother[:,2] == dz)[0]
    base_occ_indices = np.unique(np.random.choice(base_indices, n_seeds, replace = False))


    # Initialize the individual configuration in one hot encoding
    individual = np.zeros(len(mother),dtype = int)
    individual[base_occ_indices] = 1
    individual = list(individual)

    return individual

def check_two_individual_difference(ind1, ind2):

    diff_flag = True
    diff_factor = np.sum(np.abs(np.array(ind1[0]) - np.array(ind2[0])))
    if diff_factor == 0: diff_flag = False
    else: pass

    return diff_flag

def arrange_atoms(atoms):
    """
    Arrange ase object based on the chemical symbols, easier to save to CONTCAR
    """
    # Get info from atoms
    symbols = atoms.get_chemical_symbols()
    atom_positions =  atoms.get_positions()
    pbc = atoms.get_pbc()
    cell = atoms.get_cell()
    ce_positions = []
    pd_positions = []
    o_positions = [] 
    
    for si, pi in zip(symbols, atom_positions):
        if si == 'Ce': ce_positions.append(pi)
        if si == 'O': o_positions.append(pi)
        if si == 'Pd': pd_positions.append(pi)
        
    ce_positions = np.array(ce_positions)
    o_positions = np.array(o_positions)
    pd_positions = np.array(pd_positions)
    
    # Construct new atom
    atom_positions_new = np.concatenate((ce_positions,o_positions,pd_positions), axis = 0) 
    chemical_symbols_new = 'Ce'+str(len(ce_positions))+'O'+str(len(o_positions))+'Pd'+str(len(pd_positions))
    atoms_new = Atoms(chemical_symbols_new, positions = atom_positions_new, cell = cell, pbc = pbc)
    
    return atoms_new
    
    
#%%

class generator(energy.Pdn):

    def __init__(self, model_file, mother = super_mother,  super_cell_flag = True, T = 0, nseeds = 20):

        '''
        Inherit from energy.Pdn object
        '''

        super().__init__(model_file, mother, super_cell_flag)
        self.mother = mother

        # number of Pd atoms in the base layer
        self.nseeds = nseeds

        # Ensemble temperature
        self.T = T
        
        # Cutoff between large and small clusters
        self.ncutoff_0 = 8
        self.ncutoff_1 = 16
        self.ncutoff_2 = 29
    


    def occupancy():
        '''
        Creat occupancy for a node in the configuration, value between 0 or 1
        '''
        occ = random.randint(0, 1)
        return occ


    def evaluate(self, individual, individual_energy = None):
        '''
        Evaluate the fitness as energy
        '''
        # Convert individual to 1D array
        ind = np.array(individual[0])
        config = one_hot_to_index(ind)
        # fitness1 - the energy of the structure
        if individual_energy == None:
            E_pred, _ = self.predict_E(config)
        else:
            E_pred = individual_energy
        # fitness2 - the number of isolated atoms
        # fitness3 - the negative average CN
        # fitness4 - the average layer number, center of mass
        n_isolate,  floating_flag , cn_average_neg, layer_average = structure_ids(config)

        # For small clusters,
        # the focus is on center of mass and no isolate atoms
        
        if self.nseeds < self.ncutoff_0:
            return (floating_flag, n_isolate, layer_average, E_pred,  cn_average_neg)
        if (self.nseeds >= self.ncutoff_0) and (self.nseeds < self.ncutoff_1):
            return (floating_flag, n_isolate, E_pred, layer_average, cn_average_neg)
        if (self.nseeds >= self.ncutoff_1) and (self.nseeds < self.ncutoff_2):
            return (floating_flag, E_pred, n_isolate, cn_average_neg, layer_average)
        if self.nseeds >= self.ncutoff_2:
            return (floating_flag, cn_average_neg, E_pred, n_isolate, layer_average)


    def mate_cxtwopoints(self, ind1, ind2):
        '''
        pick two points and creates an interval
        exchange the interval with the two individuals
        '''
        # Set energy flag as false
        energy_flag1 = False
        energy_flag2 = False
        # Distribution factor from Boltzmann distributin
        w = 0

        # flatten 2D list
        ind1_temp = np.array(ind1[0])
        ind2_temp = np.array(ind2[0])

        # occupied nodes indices
        config1 = np.array(one_hot_to_index(ind1_temp))
        config2 = np.array(one_hot_to_index(ind2_temp))

        # Predict the energy of old structures
        E_pred1, _ = self.predict_E(config1)
        E_pred2, _ = self.predict_E(config2)

        # Swap the session between two points
        size = self.nseeds

        cxpoint1 = random.randint(0, size)
        cxpoint2 = random.randint(0, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else: # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # Check out common elements in two lists
        frac1 = config1[cxpoint1:cxpoint2]
        frac2 = config2[cxpoint1:cxpoint2]
        commonalities = list(set(frac1) - (set(frac1) - set(frac2)))
        frac1 = [fi for fi in frac1 if fi not in commonalities]
        frac2 = [fi for fi in frac2 if fi not in commonalities]

        # swap the fragment selected
        config1 = frac2 + [ci for ci in config1.copy() if ci not in frac1]
        config2 = frac1 + [ci for ci in config2.copy() if ci not in frac2]


        # Predict the energy for new structures
        E_pred1_new, _ = self.predict_E(config1)
        E_pred2_new, _ = self.predict_E(config2)

        delta_E1 = E_pred1_new - E_pred1
        delta_E2 = E_pred2_new - E_pred2
        
        # calculate the accpetance ratio
        acceptance_ratio1 = np.min([1, np.exp(-delta_E1/kb/self.T)])
        acceptance_ratio2 = np.min([1, np.exp(-delta_E2/kb/self.T)])
        raw_df_acceptance = {'ratio': [acceptance_ratio1, acceptance_ratio2], 
                              'move':   ['mate']*2}
        # Convert to a dataframe
        df_acceptance = pd.DataFrame(raw_df_acceptance)
        with open(self.filename_acceptance, 'a') as f:
            df_acceptance.to_csv(f, header=f.tell()==0)

        # accept the change if energy going downhill
        if delta_E1 <= 0:
            energy_flag1 = True
        # test using Boltzmann distribution
        else:
            if self.T > 0: w = np.exp(-delta_E1/kb/self.T)
            if np.random.rand() <= w:
                energy_flag1 = True

        # accept the change if energy going downhill
        if delta_E2 <= 0:
            energy_flag2 = True
        # test using Boltzmann distribution
        else:
            if self.T > 0: w = np.exp(-delta_E2/kb/self.T)
            if np.random.rand() <= w:
                energy_flag2 = True

        # Exchange the two individuals
        if energy_flag1:
            ind1[0] = index_to_one_hot(config1)
            E_pred1 = E_pred1_new
            self.isuccess += 1
            print("\tMating individual 1 Successful, number of nodes {}".format(len(config1)))

        if energy_flag2:
            ind2[0] = index_to_one_hot(config2)
            E_pred2 = E_pred2_new
            self.isuccess += 1
            print("\tMating individual 2 Successful, number of nodes {}".format(len(config2)))
            
        return ind1, ind2


    def mutate_metropolis(self, individual):

        '''
        Mutate an individual based on metropolis and structural rules
        '''
        # Distribution factor from Boltzmann distributin
        w = 0

        x = np.array(individual[0])
        config = one_hot_to_index(x)
        E_pred, _ = self.predict_E(config)

        # Propose a MC move
        x_new, occ_new, _ =  self.swap_occ_empty_fast(x)
        config_new = one_hot_to_index(x_new)
        E_pred_new, _  = self.predict_E(config_new)

        # #still need to check minimum distance just in case the new node is closer NN1 to other nodes
        # distance_flag = energy.check_Pd_Pd_distance(config_new, self.mother)

        # # check if the new node is NN1 to existing nodes
        # # not sure why this is still needed but when distace = True, this can be false
        # NN_flag = energy.check_Pd_Pd_neighboring(occ_new, config_new, self.mother)

        energy_flag = False
        delta_E = E_pred_new - E_pred
        
        # calculate the accpetance ratio
        acceptance_ratio = np.min([1, np.exp(-delta_E/kb/self.T)])
        raw_df_acceptance = {'ratio': [acceptance_ratio], 'move': ['mutation-single']}
        # Convert to a dataframe
        df_acceptance = pd.DataFrame(raw_df_acceptance)
        with open(self.filename_acceptance, 'a') as f:
            df_acceptance.to_csv(f, header=f.tell()==0)
        
        # accept the change if energy going downhill
        if delta_E <= 0 or self.T== np.inf :
            energy_flag = True
        # test using Boltzmann distribution
        else:
            if self.T > 0: 
                w = np.exp(-delta_E/kb/self.T)
                if np.random.rand() <= w:
                    energy_flag = True

        if (energy_flag): # and distance_flag and NN_flag):

            x = x_new
            E_pred = E_pred_new
            self.isuccess += 1
            #print("\tMutating individual Successful")

        else:
            pass

        # return the new individual
        individual[0] = list(x)

        return individual, E_pred

    def mutate_shuffle(self, individual, alpha = 1.0):

        """
        Shuffle the attributes of the input individual and return the mutant.
        :param individual: Individual to be mutated.
        :param indpb: Independent probability for each attribute to be exchanged to
                    another position. i.e how many individual nodes can be changed
        """
        # make a copy of the individual list
        individual_list = individual[0].copy()

        # Distribution factor from Boltzmann distributin
        w = 0

        # Calculate the energy of the current individual
        x = np.array(individual[0])
        config = one_hot_to_index(x)
        E_pred, _ = self.predict_E(config)

        # Shuttfle the indices
        size = len(individual_list)
        for i in range(size):
            if random.random() < alpha:
                swap_indx = random.randint(0, size - 2)
                if swap_indx >= i:
                    swap_indx += 1
                individual_list[i], individual_list[swap_indx] = \
                    individual_list[swap_indx], individual_list[i]

        # Calculate new energy
        x_new = np.array(individual_list)
        config_new = one_hot_to_index(x_new)
        E_pred_new, _  = self.predict_E(config_new)

        energy_flag = False
        delta_E = E_pred_new - E_pred

        # accept the change if energy going downhill
        if delta_E <= 0:
            energy_flag = True
        # test using Boltzmann distribution
        else:
            if self.T > 0: w = np.exp(-delta_E/kb/self.T)
            if np.random.rand() <= w:
                energy_flag = True

        if energy_flag: 
            individual[0] = individual_list
            E_pred = E_pred_new
            self.isuccess += 1
            #print("\tMutating individual Successful")

        else: pass

        return individual, E_pred

    def mutate_swap_iso_neighbors(self, individual, alpha = 1.0):

        '''
        Mutate an individual based on metropolis and swapping rules
        '''
        # Distribution factor from Boltzmann distributin
        w = 0

        x = np.array(individual[0])
        config = one_hot_to_index(x)
        E_pred, _ = self.predict_E(config)

        # Propose a MC move
        x_new, _, _ =  self.swap_iso_neighbors(x)
        config_new = one_hot_to_index(x_new)
        E_pred_new, _  = self.predict_E(config_new)

        #still need to check minimum distance just in case the new node is closer NN1 to other nodes
        #distance_flag = energy.check_Pd_Pd_distance(config_new, self.mother)

        # check if the new node is NN1 to existing nodes
        # not sure why this is still needed but when distace = True, this can be false
        #NN_flag = energy.check_Pd_Pd_neighboring(occ_new, config_new, self.mother)

        energy_flag = False
        delta_E = E_pred_new - E_pred
        
        # calculate the accpetance ratio
        acceptance_ratio = np.min([1, np.exp(-delta_E/kb/self.T)])
        raw_df_acceptance = {'ratio': [acceptance_ratio], 'move': ['mutation-multi']}
        # Convert to a dataframe
        df_acceptance = pd.DataFrame(raw_df_acceptance)
        with open(self.filename_acceptance, 'a') as f:
            df_acceptance.to_csv(f, header=f.tell()==0)

        # accept the change if energy going downhill
        if delta_E <= 0:
            energy_flag = True
        # test using Boltzmann distribution
        else:
            if self.T > 0: w = np.exp(-delta_E/kb/self.T)
            if np.random.rand() <= w:
                energy_flag = True

        if (energy_flag): #and distance_flag and NN_flag):

            x = x_new
            E_pred = E_pred_new
            self.isuccess += 1
            print("\tMutating multi Successful")

        else:
            pass

        # return the new individual
        individual[0] = list(x)

        return individual, E_pred

    def select_diversity_check(self, population = None, population_energy = None):
        '''
        Select and sort unique individuals in the population 
        based on their energy
        disgard the rest to keep the diversity
        '''
        energy_unique, indices_unique = np.unique(population_energy, return_index=True)
        population_unique = [population[i] for i in indices_unique]
        energy_unique = list(energy_unique)

        return population_unique, energy_unique


    def make_initial_population(self, COMM = None, toolbox = None, n = None):

        rank = get_rank(COMM)
    #    if rank == 0:
        print('\t{}  Core {}  Building initial population'.format(get_time(), rank))
        population = toolbox.population(n = n)
    #    else:
    #        population = None

        return population


    def evaluate_population(self, COMM = None, toolbox = None, population = None, population_energy = None):
        rank = get_rank(COMM)
        size = get_size(COMM)
        
    #    if rank == 0:
    #            jobs_split = np.array_split(range(len(population)), size)
    #            population_split = []
    #            for jobs in jobs_split:
    #                x = []
    #                for job in jobs:
    #                    x.append(population[job])
    #                population_split.append(x)
    #            print('\t{}  Core {}  Distributing individuals'.format(get_time(), rank))
    #    else:
    #        population_split = None
    #        jobs_split = None
    #
    #    if COMM is None:
        population_mpi = population
        jobs_mpi = range(len(population))
    #    else:
    #        population_mpi = COMM.scatter(population_split, root = 0)
    #        jobs_mpi = COMM.scatter(jobs_split, root = 0)

        #Evaluate fitness
        fitnesses_list = []
        fitnesses_mpi = {}
        for i, individual_mpi in zip(jobs_mpi, population_mpi):
            if population_energy  == None: individual_energy = None
            else: individual_energy = population_energy[i]
            fitnesses_mpi[i] = toolbox.evaluate(individual_mpi, individual_energy)
        print('\t{}  Core {}  Finished evaluating individuals'.format(get_time(), rank))
        #if COMM is None:
        fitnesses_list = [fitnesses_mpi]
    #    else:
    #        fitnesses_list = MPI.COMM_WORLD.gather(fitnesses_mpi, root = 0)
    #    if rank == 0:
        print('\t{}  Core {}  Assigning fitness to population.'.format(get_time(), rank))
        for fitnesses_dict in fitnesses_list:
            for i, fitness in fitnesses_dict.items():
                population[i].fitness.values = fitness
    #    else:
    #        population = None

        return population

    def write_initial_population(self, COMM = None, toolbox = None, population = None, history_flag = True, best_flag = True, acceptance_flag = True):

        '''
        Function to create a csv file and save all individuals
        '''
        # the list saving all individuals in history
        self.history = population.copy()
        generation_no  = 0 # the generation number
        self.isuccess = 0 # the number of successful event
        fitnesses_initial = self.get_fitnesses(population)
        i_gen_array = np.ones(len(population)) * generation_no
        
        if history_flag:
            # Write history output
            
            self.filename_history = 'ga_history_output_' + str(self.nseeds)+ '_' + str(self.T) + 'k' +  '.csv'
            print('\tWriting generation {} to history_output.csv'.format(generation_no))
            # create a dictionary first
            raw_df = {'generation no.': i_gen_array,
                        'fitness1': fitnesses_initial[:,0],
                        'fitness2': fitnesses_initial[:,1],
                        'fitness3': fitnesses_initial[:,2],
                        'fitness4': fitnesses_initial[:,3],
                        'fitness5': fitnesses_initial[:,4],
                        'individual': population
                        }
            # Convert to a dataframe
            df = pd.DataFrame(raw_df)
            # Delete the file generated previously
            if os.path.exists(self.filename_history):  os.remove(self.filename_history)
            with open(self.filename_history, 'a') as f:
                df.to_csv(f, header=f.tell()==0)
        
        if best_flag:
            # Write generation best output
            self.filename_best = 'ga_generation_best_' + str(self.nseeds) +  '_' + str(self.T) + 'k' + '.csv'
            generation_best = self.find_best_individual_k(toolbox, population, k = 1)
            fitnesses_best = self.get_fitnesses(generation_best)
            print('\tWriting generation {} to generation_best_output.csv'.format(generation_no))
            print( '\tIndividual with best fitness: Fitness = {} '.format(generation_best[0].fitness.values))
            raw_df_best = {'generation no.': generation_no,
                        'fitness1': fitnesses_best[0,0],
                        'fitness2': fitnesses_best[0,1],
                        'fitness3': fitnesses_best[0,2],
                        'fitness4': fitnesses_best[0,3],
                        'fitness5': fitnesses_best[0,4],
                        'individual': generation_best
                        }

            # Convert to a dataframe
            df_best = pd.DataFrame(raw_df_best)
            # Delete the file generated previously
            if os.path.exists(self.filename_best):  os.remove(self.filename_best)
            with open(self.filename_best, 'a') as f:
                df_best.to_csv(f, header=f.tell()==0)
                
        if acceptance_flag:
            # write acceptance ratio and move type
            self.filename_acceptance =  'ga_move_acceptance_' + str(self.nseeds) +  '_' + str(self.T) + 'k' + '.csv'
            # Delete the file generated previously
            if os.path.exists(self.filename_acceptance):  os.remove(self.filename_acceptance)
            raw_df_acceptance= {'ratio': [0.0], 'move': ['initial' ]}
            # Convert to a dataframe
            df_acceptance = pd.DataFrame(raw_df_acceptance)
            with open(self.filename_acceptance, 'a') as f:
                df_acceptance.to_csv(f, header=f.tell()==0)
            

    def write_history(self, COMM = None, population = None, generation_no = None, history_flag = True, best_flag = True):

        # append to a history list contains all generations
        for pi in population:
            self.history.append(pi)
        # update the generation number
        i_gen_array = np.ones(len(population)) * generation_no
        fitnesses = self.get_fitnesses(population)
        
        if history_flag:
            print('\tWriting generation {} to history_output.csv'.format(generation_no))


            # create a dictionary first
            raw_df = {'generation no.': i_gen_array,
                        'fitness1': fitnesses[:,0],
                        'fitness2': fitnesses[:,1],
                        'fitness3': fitnesses[:,2],
                        'fitness4': fitnesses[:,3],
                        'fitness5': fitnesses[:,4],
                        'individual': population
                        }
            # Convert to a dataframe
            df = pd.DataFrame(raw_df)
            
            with open(self.filename_history, 'a') as f:
                df.to_csv(f, header=f.tell()==0)

        if best_flag:
            # Write generation best output
            print('\tWriting generation {} to generation_best_output.csv'.format(generation_no))
             # Print the stats for the best individual after sorted
            print( '\tIndividual with best fitness: Fitness = {} '.format(population[0].fitness.values))
            raw_df_best = {'generation no.': generation_no,
                        'fitness1': fitnesses[0,0],
                        'fitness2': fitnesses[0,1],
                        'fitness3': fitnesses[0,2],
                        'fitness4': fitnesses[0,3],
                        'fitness5': fitnesses[0,4],
                        'individual': population[0]
                        }

            # Convert to a dataframe
            df_best = pd.DataFrame(raw_df_best)

            with open(self.filename_best, 'a') as f:
                df_best.to_csv(f, header=f.tell()==0)


    def generate_offspring(self, COMM = None, toolbox = None, population = None, cxpb = None):
        rank = get_rank(COMM)
    #    if rank == 0:
        print( '\t{}  Core {}  Generating offspring'.format(get_time(), rank))
        offspring = [toolbox.clone(individual) for individual in population]
        
        if len(population) > 1:
            #Apply crossover and mutation on the offspring
            #print '\tMaking children'
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    #skip mating if two individuals are identical
                    if check_two_individual_difference(child1, child2):
                        child1, child2 = toolbox.mate(ind1 = child1, ind2 = child2)
                        del child1.fitness.values
                        del child2.fitness.values
       
        return offspring

    def mutate_offspring(self, COMM = None, toolbox = None, population = None):
        rank = get_rank(COMM)
    #    if rank == 0:
        print( '\t{}  Core {}  Mutating offspring metropolis'.format(get_time(), rank))
        #print '\tApplying mutations'
        population_energy = []
        for mutant in population:
            mutant, energy = toolbox.mutate(mutant)
            population_energy.append(energy)
            del mutant.fitness.values
            
    #    else:
    #        population = None
        return population, population_energy

    def mutate_offspring_multi(self, COMM = None, toolbox = None, population = None):
        rank = get_rank(COMM)
    #    if rank == 0:
        print( '\t{}  Core {}  Mutating offspring multiple'.format(get_time(), rank))
        population_energy = []
        for mutant in population:
            mutant, energy = toolbox.mutate_multi(mutant)
            population_energy.append(energy)
            del mutant.fitness.values
    #    else:
    #        population = None
        return population, population_energy

    def make_next_population(self, COMM = None, toolbox = None,  population = None, offspring = None, npop= None, nremain = None):
        rank = get_rank(COMM)
    #    if rank == 0:
        print( '\t{}  Core {}  Generating new population'.format(get_time(), rank))
        # Select the fittest k unique offsprings
        # Select best 
        candidates_sorted = self.find_best_individual_k(toolbox, offspring + population, k = 2*npop)
        candidates_energy = self.get_fitnesses(candidates_sorted)[:,1]
        candidates_unique, _ = self.select_diversity_check(candidates_sorted, candidates_energy)
            
        
        if len(candidates_unique) < nremain:
            candidates_remain = candidates_unique
        else:  # Select fittest n from the candidates
            candidates_remain = candidates_unique[:nremain]
        # add n-k new individuals 

        j = npop - len(candidates_remain)
        if j > 0 :
            newcomer = toolbox.population(n = j)
            newcomer = self.evaluate_population(COMM, toolbox, newcomer)
            # Assign back to population
            population[:] =  candidates_remain + newcomer
        else: 
            population[:] = candidates_remain

        return population


    def calculate_statistics(self, COMM = None, population = None, generation_no = None):
        rank = get_rank(COMM)
    #    if rank == 0:
        print( '\t{}  Core {}  Calculating statistics'.format(get_time(), rank))
        fitnesses_all = self.get_fitnesses(population)
        # Only extract energy 
        if self.nseeds < self.ncutoff_0:
            energy_index = 3
        if (self.nseeds >= self.ncutoff_0) and (self.nseeds < self.ncutoff_1):
            energy_index = 2
        if self.nseeds >= self.ncutoff_1 and (self.nseeds < self.ncutoff_2):
            energy_index = 1
        if self.nseeds >= self.ncutoff_2: #and (self.nseeds < self.ncutoff_1):
            energy_index = 2
        
        fitnesses = fitnesses_all[:,energy_index]
        avg = np.mean(fitnesses)
        sd = np.std(fitnesses)
        min_val = np.min(fitnesses)
        max_val = np.max(fitnesses)

        self.filename_stats = 'ga_stats_output_' + str(self.nseeds)+ '_' + str(self.T) + 'k' + '.csv'
        print('\tWriting generation {} to stats_output.csv'.format(generation_no))
        # create a dictionary first
        raw_df = {'generation no.': [generation_no],
                    'mean': [avg],
                    'sd': [sd],
                    'min_val': [min_val],
                    'max_val': [max_val]
                    }  
        # Convert to a dataframe
        df = pd.DataFrame(raw_df)
        if generation_no == 0:
            # Delete the file generated previously
            if os.path.exists(self.filename_stats):  os.remove(self.filename_stats)
        with open(self.filename_stats, 'a') as f:
            df.to_csv(f, header=f.tell()==0)

    def print_generation_number(self, COMM = None, generation = None):
        rank = get_rank(COMM)
    #    if rank == 0:
        print( '{}  Core {}  Generation {}'.format(get_time(), rank, generation))

    def get_fitnesses(self, population = None):
        '''
        normalize the fitness values
        '''
        fitness_2d_array = np.array([individual.fitness.values for individual in population])

        #fitness_max = np.max(fitness_tuple, axis = 0)
        #normalized_fit = fitness_tuple/fitness_max
        #normalized_fit[:,-1] = 1-normalized_fit[:,-1]

        return fitness_2d_array

    def find_best_individual_k(self,  toolbox = None, population = None, k = None, fit_attr="fitness"):
        '''
        make a clone and return the sorted population
        No index were returned here
        '''
        pop_copy = [toolbox.clone(individual) for individual in population]

        return sorted(pop_copy, key=attrgetter(fit_attr), reverse=True)[:k]
        

    def find_best_individual(self, COMM = None, population = None, nbest = 1):
        '''
        find the best individual in a population
        '''
        #rank = get_rank(COMM)
    #    if rank == 0:
        # get fitness as a 2D array (npop * features)
        fitnesses = self.get_fitnesses(population)

        fiv = []
        for fi in range(fitnesses.shape[1]):
            fiv.append(fitnesses[:,fi])

        if fitnesses.shape[1] == 1:

            ind = np.argsort(fiv[0])
            best_i = ind[0]
        else:
            ind = np.lexsort((fiv[-1], fiv[-2], fiv[-3], fiv[-4], fiv[-5]))
            best_i = ind[0]

        print( '\tIndividual with best fitness:')
        print( '\tFitness = {} '.format(population[best_i].fitness.values))
        #print( '\tCV RMSE = {} eV'.format(np.sqrt(population[i].fitness.values[0])))
        return best_i



    def hall_of_fame(self, COMM= None, nbest = 500, print_flag = True):
        '''
        Evaluate the fitness of entire population
        Print out the best
        Write the fittest nbest to a cvs file
        '''
        history = self.history.copy()
        fitnesses = self.get_fitnesses(history)

        # Take out the repeated individuals in the population
        fitnesses_unique, indices_unique = np.unique(fitnesses, axis=0, return_index=True)
        history_unique = [history[i] for i in indices_unique]

        # Evaluate based on fitness
        fiv = []
        for fi in range(fitnesses_unique.shape[1]):
            fiv.append(fitnesses_unique[:,fi])

        if fitnesses_unique.shape[1] == 1:
            ind = np.argsort(fiv[0])
        else:
            ind = np.lexsort((fiv[-1], fiv[-2], fiv[-3], fiv[-4], fiv[-5]))

        # Select nbest individuals
        if nbest > len(history_unique): nbest = len(history_unique)

        best_indices = np.array(ind[:nbest])
        history_best =  [history_unique[i] for i in best_indices]
        fitnesses_unique_best = fitnesses_unique[ind[:nbest]]

        # Save hall of fame into a cvs file
        # create a dictionary first
        raw_df = {'individual rank': np.arange(nbest),
                    'fitness1': fitnesses_unique_best[:,0],
                    'fitness2': fitnesses_unique_best[:,1],
                    'fitness3': fitnesses_unique_best[:,2],
                    'fitness4': fitnesses_unique_best[:,3],
                    'fitness5': fitnesses_unique_best[:,4],
                    'individual': history_best
                    }
        # Convert to a dataframe
        df = pd.DataFrame(raw_df)
        filename = 'ga_hall_of_fame_' + str(self.nseeds) +  '_' + str(self.T) + 'k' + '.csv'
        df.to_csv(filename)

        # Print the output
        if print_flag:
            print('\nTotal Successful Events: {}'.format(self.isuccess))
            print( 'Hall of Fame top {}:'.format(nbest))
            for i, hi in enumerate(history_best):
                print( '\tIndividual {} with best fitness:  Fitness = {}'.format(i, hi.fitness.values))

        return history_best

    def winner_details(self, COMM = None, population = None, ranked_flag = True):
        '''
        Given a ranked population based on fitness
        return the top individal and its configuration
        '''
        if not ranked_flag:
            i = self.find_best_individual(COMM, population)
        else: i = 0

        best_ind = population[i]
        best_fitness = best_ind.fitness.values
        best_config = one_hot_to_index(best_ind[0])

        return(best_ind, best_fitness, best_config)

