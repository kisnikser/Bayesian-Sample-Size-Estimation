import numpy as np
import random
from chromosome import Chromosome

def traversal(chromosome, index):
    """
    Function to traverse the tree from the given index
    @param: index - start position
    @chromosome: chromosome to be traversed
    """
    if chromosome.gen[index] == 'x' or chromosome.gen[index] == '1':
        return index + 1
    elif chromosome.gen[index] in chromosome.func_set[1]:
        return traversal(chromosome, index + 1)
    else:
        index_new = traversal(chromosome, index + 1)
        return traversal(chromosome, index_new)

def mutate(chromosome):
    """
    Function to mutate a chromosome
    @param: chromsome - chromosome to be mutated
    @return: the mutated chromosome
    """
    index = np.random.randint(len(chromosome.gen))
    
    if chromosome.gen[index] in chromosome.func_set[1]:
        chromosome.gen[index] = random.choice(chromosome.func_set[1])
    elif chromosome.gen[index] in chromosome.func_set[2]:
        chromosome.gen[index] = random.choice(chromosome.func_set[2])
    else:
        chromosome.gen[index] = random.choice(['x', '1'])
    
    return chromosome

def selection(population, num_sel):
    """
    Function to select a member of the population for crossing over
    @param: population - population of chromosomes
    @param: num_sel - number of chromosome selected from the population
    @return: the selected chromosome
    """
    sample = random.sample(population.list, num_sel)
    best = sample[0]
    for i in range(1, len(sample)):
        if population.list[i].fitness < best.fitness:
            best = population.list[i]
    
    return best

def cross_over(mother, father, max_depth):
    """
    Function to cross over two chromosomes in order to obtain a child
    @param mother: - chromosome
    @param father: - chromosome
    @param max_depth - maximum_depth of a tree
    """
    child = Chromosome(mother.func_set, mother.depth, None)
    start_m = np.random.randint(len(mother.gen))
    start_f = np.random.randint(len(father.gen))
    end_m = traversal(mother, start_m)
    end_f = traversal(father, start_f)
    child.gen = mother.gen[:start_m] + father.gen[start_f:end_f] + mother.gen[end_m:]
    if child.get_depth() > max_depth: #and random.random() > 0.2:
        child = Chromosome(mother.func_set, mother.depth)
    return child


def get_best(population):
    """
    Function to get the best chromosome from the population
    @param: population to get the best chromosome from
    @return: best chromosome from population
    """
    best = population.list[0]
    for i in range(1, len(population.list)):
        if population.list[i].fitness < best.fitness: # fitness чем меньше, тем лучше!
            best = population.list[i]
    
    return best

def get_worst(population):
    """
    Function to get the worst chromosome of the population
    @param: population - 
    @return: worst chromosome from the population
    """
    worst = population.list[0]
    for i in range(1, len(population.list)):
        if population.list[i].fitness > worst.fitness:
            worst = population.list[i]
    
    return worst

def replace_worst(population, chromosome):
    """
    Function to change the worst chromosome of the population with a new one
    @param: population - population 
    @param: chromosome - chromosome to be added
    """
    worst = get_worst(population)
    if chromosome.fitness < worst.fitness:
        for i in range(len(population.list)):
            if population.list[i].fitness == worst.fitness:
                population.list[i] = chromosome
                break
    return population

def roulette_selecion(population):
    """
    Function to select a member of the population usingq roulette selection
    @param: population - population to be selected from
    """
    fitness = [chrom.fitness for chrom in population.list]
    order = [x for x in range(len(fitness))]
    order = sorted(order, key=lambda x: fitness[x])
    fs = [fitness[order[i]] for i in range(len(fitness))]
    sum_fs = sum(fs)
    max_fs = max(fs)
    min_fs = min(fs)
    p = random.random()*sum_fs
    t = max_fs + min_fs
    choosen = order[0]
    for i in range(len(fitness)):
        p -= (t - fitness[order[i]])
        if p < 0:
            choosen = order[i]
            break
    return population.list[choosen]