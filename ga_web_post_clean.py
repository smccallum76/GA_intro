# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 07:18:24 2019

@author: Scott McCallum
"""

import pandas as pd
from scipy import stats
import numpy as np
import ga_web_post_functions_clean as gwp

''' ---------------------------------------------------------------- '''
''' LINEAR REGRESSION MODEL '''
''' ---------------------------------------------------------------- '''
data = pd.read_csv('input\\stattler_rhob_toc.csv')

slope, intercept, r_value, p_value, std_err = stats.linregress(data.RHOB, data.TOC)

line_fit = slope*data.RHOB + intercept
#residual sum of squares
rss = np.sum((line_fit - data.TOC)**2, axis=0)

# convert values to strings for plotting annotations
slope_str = str(round(slope, 3))
int_str = str(round(intercept, 3))
r2_str = str(round(r_value**2, 3))
rss_str = str(round(rss, 3))

''' ---------------------------------------------------------------- '''
''' GENETIC ALGORTIHM SETTINGS (user defined) '''
''' ---------------------------------------------------------------- '''
# make a parent matrix  and convert each to a vector
pop_size = 200 # number of individuals in population
num_parents = int(pop_size/2)
generations = 200
mutation_rate = 50 # of rows (chromosomes) that will have one gene mutated
max_slope = 0
min_slope = -100
max_int = 200
min_int = 0

# couple items to track for plotting fitness
best_parents_mean = [] # keep this to use for plotting
best_parents_min =[] # keep this to retain the best model from each generation
''' ---------------------------------------------------------------- '''
'''INITIAL PARENTS '''
parents = gwp.starting_population(min_slope, max_slope, min_int, max_int, pop_size)
''' ---------------------------------------------------------------- '''

''' ---------------------------------------------------------------- '''
''' OPTIMIZATION LOOP '''
''' ---------------------------------------------------------------- '''
for g in range(generations):
    ''' FITNESS '''
    best_parents, best_parents_fitness = gwp.fitness(parents, num_parents, data)
    # some stuff to plot
    best_parents_mean.append(best_parents_fitness.Fit.mean())
    best_parents_min.append(best_parents_fitness.iloc[0,:])
    running_fitness = best_parents_fitness.loc[0,'Fit']
    if g%10==0:
        print('generation:', g, '     fitness: ', running_fitness)
    ''' GENE SPLITTER '''
    par_strings = gwp.gene_splitter(best_parents)
    ''' CROSSOVER '''
    offspring = gwp.crossover(best_parents, par_strings)
    ''' GENE SPLICER '''
    next_gen = gwp.gene_splicer(best_parents, offspring)
    ''' MUTATION AND NEW PARENT POPULATION '''
    parents = gwp.mutation(mutation_rate, num_parents, next_gen)
    
''' ---------------------------------------------------------------- '''


        

    







