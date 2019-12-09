# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:24:15 2019

@author: Scott McCallum
"""
import pandas as pd
import numpy as np

def starting_population(min_slope, max_slope, min_int, max_int, pop_size):
    # The starting population is built specifically to the problem being 
    # optimized.  In this example only a slope and intercept gene are needed
    # so the initial population will be pop_size X 2.  
    parents = np.random.rand(pop_size,2)
    # random value for the slope
    min_max_slope = np.random.randint(low=min_slope, high=max_slope, size=(pop_size,1))
    # random value for the intercept
    min_max_int = np.random.randint(low=min_int, high=max_int, size=(pop_size,1))
    # populate the parent array
    parents[:,0] = parents[:,0] * min_max_slope[:,0]
    parents[:,1] = parents[:,1] * min_max_int[:,0]
    # add some column names to make life a little easier
    parents = pd.DataFrame(parents, columns=['slope', 'int'])
    return parents

def fitness(parents, num_parents, data):
    # measure the fitness of each individual parent (chromosome) in the parent
    # population.  
    fit_list=[] # list of each parents fitness
    for i in range(len(parents)):
        # prediction
        prediction = parents.loc[i, 'slope']*data.RHOB + parents.loc[i, 'int']
        # fitness based on residual sum of squres
        fitness = np.sum((prediction - data.TOC)**2, axis=0)
        fit_list.append(fitness)
        
    # adding the fitness value to the parents array
    parents['Fit'] = fit_list
    # sort with lowest error (best fitness) values at the top of the list
    parents.sort_values(by=['Fit'], inplace=True)
    parents.reset_index(inplace=True, drop=True)
    # copy of the upper half of most fit parents that will be used for 
    # reproduction
    best_parents = parents.iloc[0:num_parents,0:-1] 
    best_parents_fitness = parents.iloc[0:num_parents,:]
    return best_parents, best_parents_fitness

def gene_splitter(best_parents):
    # splitting the slope and intercept into two new genes that include the 
    # integer and decimal component of both the slope and intercept.  This 
    # increases the gene count from 2 to 4 genes per chromosome.  Typically 
    # this isn't necessary, but with only two genes in this function it was
    # found that this approach introduced increased variance.  
    # ex: slope = 70.123 will become:
    #   slp_10 = 70
    #   slp_dec = 123
    # same idea for the intercept.  
    
    # round and convert to string
    # note: par is abbreviation for parents
    par_slopes = round(best_parents['slope'], 5).astype(str)
    par_inter = round(best_parents['int'], 5).astype(str)
    # split on the decimal
    par_slopes = par_slopes.str.split('.',n=1, expand=True)
    par_inter = par_inter.str.split('.',n=1, expand=True)
    # make an empty df
    par_strings = pd.DataFrame()
    # populate df with the split values
    par_strings['slp_10'] = par_slopes[0]
    par_strings['slp_dec'] = par_slopes[1]
    par_strings['int_10'] = par_inter[0]
    par_strings['int_dec'] = par_inter[1]
    return par_strings

def crossover(best_parents, par_strings):
    # credit to Ahmed Gad for much of the crossover component.  Some 
    # modifications were made for the purposes of this example.
    # https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
    
    # empty array of offspring of the same size as par_strings
    offspring = np.zeros(shape=np.shape(par_strings))
    for k in range(len(best_parents)):
        # crossover point is often just the midpoint, but in this modification
        # the crossover point is randomly selected.  
        crossover = int(np.random.uniform(low=0,high=np.shape(par_strings)[1]))
        # idx of the first parent to be used in crossover
        parent1_idx = k%best_parents.shape[0]
        # idx of the second parent to be used in crossover
        parent2_idx = (k+1)%best_parents.shape[0]
        # make an initial copy of parent k
        offspring[k,:] = par_strings.iloc[parent1_idx, :]
        #offspring[k,0:crossover] = par_strings.iloc[parent1_idx, 0:crossover]
        # Now replace one of the 4 (in this case) genes with a gene from 
        # the other parent.  This was a modification to crossover in that just
        # one gene was swapped instead of all the genes from some point to
        # the end.  The improvement to the model was minor if at all, and it 
        # could be argued that it should not be used.  A more traditional 
        # approach is included, just uncomment the offspring lines above and
        # below (lines 92 and 101)
        offspring[k, crossover] = par_strings.iloc[parent2_idx, crossover]
        #offspring[k, crossover::] = par_strings.iloc[parent2_idx, crossover::]
    return offspring
    
def gene_splicer(best_parents, offspring):
    # basically the same comments as for the gene_splitter, but now the parts
    # are being put back together so they can be fed to the next generation. 
    
    # bring the offspring back into a 2 column matrix of float values
    next_gen = np.array(best_parents) # offspring will be appended to this
    # the int to string step was needed due to the fact that the values had
    # been stored as float values.  
    offspring = pd.DataFrame(offspring).astype(int).astype(str)
    # combine the integer and decimal components for slope and intercept
    slope_nums = offspring[0] + '.' + offspring[1]
    intercept_nums = offspring[2] + '.' + offspring[3]
    # convert the combined values from string to float
    slope_nums = slope_nums.astype(float)
    intercept_nums = intercept_nums.astype(float)
    # transpose was needed
    offspring = np.array([slope_nums, intercept_nums]).T
    # stick together the best parents and their new offspring into one array
    next_gen = np.append(next_gen, offspring, axis=0)
    return next_gen

def mutation(mutation_rate, num_parents, next_gen):
    ''' MUTATION '''
    if mutation_rate > 0:
        for j in range(mutation_rate):
            # index of row that will be mutated
            mut_row = int(np.random.uniform(low=num_parents, 
                                            high=np.shape(next_gen)[0]))
            # index of column that will be mutated
            mut_col = int(np.random.uniform(low=0, high=np.shape(next_gen)[1]))
            # mutation value (partially dependent on existing values)
            # this is not necessarily the best way to apply mutation, but 
            # since the ranges were so large for this problem it was unlikely 
            # that mutation would prove beneficial.  For this reason, the
            # mutation was a randomly applied multiplier (from 0.8 - 1.2) that
            # was used to adjust a random row and column based on the existing
            # value.  
            mut_value = np.random.uniform(low=0.8, high=1.2)
            next_gen[mut_row, mut_col] = next_gen[mut_row, mut_col] * mut_value
            parents = pd.DataFrame(next_gen, columns=['slope', 'int'])
    else: # this allows the used to see the impact of 0 mutation.  
        parents = pd.DataFrame(next_gen, columns=['slope', 'int'])
    return parents

