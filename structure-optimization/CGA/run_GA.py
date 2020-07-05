'''
Run GA functions on strutural optimization
Aug 10th 2019
@author: Yifan Wang
'''


import os
import sys
import pandas as pd
import numpy as np
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
from GA_functions import get_time




'''
Test_GA version 3
Test on Pd20
'''

try:
    from mpi4py import MPI
except:
    COMM = None
    rank = 0
else:
    COMM = MPI.COMM_WORLD
    rank = COMM.rank

# Seed the random
rseed = 1
np.random.seed(rseed)

T = 300 # Ensemble temperature
nseeds = 20 # Ensemble size
GA = GA_f.generator(lasso_file, mother=super_mother, super_cell_flag=True, T=T, nseeds=nseeds)

# %%
# Genetic algorithm Hyperparameters

npop = 5  # Size of population
ngen = 10  # Number of generations
nfitness = 5  # the number of fitness attribute
cxpb = 1.0  # The probability of mating two individuals
nremain = npop   # the number of individual selected in each iteration
#tournsize = 10


# Outfiles specification
history_flag = True
best_flag = True


#%% Start the simulation
start_time = time.time()
tools, base, creator = GA_f.import_deap(nfitness)
print('\n\n -----\tGenetic algorithm for structural optimization\t-----')
print('Pd{} nanocluster'.format(nseeds))
print('\n\t{}  Core {}  Reading files'.format(GA_f.get_time(), rank))


# Create the toolbox
toolbox = base.Toolbox()
# Attribute generator
toolbox.register("init_gen", GA_f.initial_individual, n_seeds=nseeds)
# the ind is a two 2D list
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.init_gen, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", GA.mate_cxtwopoints)
toolbox.register("mutate", GA.mutate_metropolis)
toolbox.register("mutate_reverse", GA.mutate_metropolis_reverse)
toolbox.register("mutate_multi", GA.mutate_swap_iso_neighbors, alpha=1.0)
#toolbox.register("select", GA.select_diversity_check)
toolbox.register("evaluate", GA.evaluate)



population = GA.make_initial_population(COMM, toolbox, npop)
population = GA.evaluate_population(COMM, toolbox, population)
GA.write_initial_population(COMM, toolbox, population, history_flag=history_flag, best_flag=best_flag)
GA.calculate_statistics(COMM, population, 0)

#  Would like to test on two individuals using two different methods
ind1 = population[0]
fit1 = population[0].fitness.values



#%%
for generation in range(ngen):
    GA.print_generation_number(COMM, generation)
    offspring = GA.generate_offspring(COMM, toolbox, population, cxpb)  # mate
    offspring, offspring_energy = GA.mutate_offspring(COMM, toolbox, offspring)  # mutate - metropolis
    offspring, offspring_energy = GA.mutate_offspring_multi(COMM, toolbox, offspring)  # mutate
    offspring, offspring_energy = GA.mutate_offspring_reverse(COMM, toolbox, offspring)  # mutate - reverse
    offspring = GA.evaluate_population(COMM, toolbox, offspring, offspring_energy)  # evaluate
    population = GA.make_next_population(COMM, toolbox,population, offspring, npop, nremain)  # duplicate, select the best k out of population+offspring
    # GA.find_best_individual(COMM, population) # report the best, should log the best here
    GA.write_history(COMM, population, generation+1, history_flag=history_flag, best_flag=best_flag)  # save to history_output
    GA.calculate_statistics(COMM, population, generation+1)

#%%
history_best = GA.hall_of_fame(COMM)
(best_ind, best_fitness, best_config) = GA.winner_details(COMM, history_best)

end_time = time.time()
min_time = (end_time - start_time) / 60

print('\nThe simulation takes {0:.4f} minutes'.format(min_time))
