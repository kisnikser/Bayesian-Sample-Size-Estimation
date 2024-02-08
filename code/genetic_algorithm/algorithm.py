from ga_operations import *

class Algorithm:
    """
    Class representing the algorithm
    """
    def __init__(self, population, iterations, datasets=None, X=None, y=None, epoch_feedback=500):
        """
        Constructor for Alrogithm class
        @param: population - population for the current algorithm
        @param: iterations - number of iterations for the algorithm
        @param: X - inputs
        @param: y - outputs
        @param: epoch_feedback - number of epochs to show feedback
        """
        self.population = population
        self.iterations = iterations
        self.datasets = datasets
        self.X = X
        self.y = y
        self.epoch_feedback = epoch_feedback
    
    def __one_step(self):
        """
        Function to do one step of the algorithm 
        """
        mother = selection(self.population, self.population.num_selected)
        father = selection(self.population, self.population.num_selected)
        #mother = roulette_selecion(self.population)
        #father = roulette_selecion(self.population)
        child = cross_over(mother, father, self.population.max_depth)
        child = mutate(child)
        child.calculate_fitness(datasets=self.datasets, X=self.X, y=self.y)
        self.population = replace_worst(self.population, child)

    def train(self):
        print("Training process has started")
        print("============================")
        for i in range(len(self.population.list)):
            self.population.list[i].calculate_fitness(datasets=self.datasets, X=self.X, y=self.y)
        for i in range(1, self.iterations+1):
            if i % self.epoch_feedback == 0:
                best_so_far = get_best(self.population)
                print(f"[Epoch {i}/{self.iterations}]")
                print(f"- Best function: {best_so_far}")
                print(f"- Best fitness: {best_so_far.fitness}")
            self.__one_step()
        return get_best(self.population)

        