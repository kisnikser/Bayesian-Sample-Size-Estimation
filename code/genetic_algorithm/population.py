import random
from chromosome import Chromosome

class Population:
    """
    Class for representing a population of chromosomes
    """

    def __init__(self, size, num_selected, func_set, depth, max_depth):
        """
        Constructor for population class
        @param: size - number of members in the population
        @param: func_set - set of functions for the population
        @param: num_selected - number of chromosomes selected from the population
        @param: depth - initial depth of a tree
        @param: max_depth - maximum depth of a tree
        """
        self.size = size
        self.num_selected = num_selected
        self.list = self.create_population(self.size, func_set, depth)
        self.max_depth = max_depth

    def create_population(self, number, func_set, depth):
        pop_list = []
        for i in range(number):
            if random.random() > 0.5:
                pop_list.append(Chromosome(func_set, depth, 'grow'))
            else:
                pop_list.append(Chromosome(func_set, depth, 'full'))
        return pop_list