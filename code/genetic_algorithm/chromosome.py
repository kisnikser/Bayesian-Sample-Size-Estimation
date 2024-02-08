import random
import math
import numpy as np
from scipy.optimize import minimize
#import warnings
#warnings.filterwarnings("error")

class Chromosome:
    """
    Class for representing a chromosome
    """
    def __init__(self, funct_set, depth, method='full'):
        """
        Constructor for Chromosome class
        @param: depth - tree depth
        @param: method - method to generate the tree, default is full
        @param: funct_set - set of functions
        """
        self.depth = depth
        self.gen = []
        self.func_set = funct_set
        self.fitness = None
        if method == 'grow':
            self.grow()
        elif method == 'full':
            self.full()

    def full(self, level=0):
        """
        Function to generate a tree in a full manner
        Every node will have exactly two children
        return: None
        """
        if level == self.depth:
            if random.random() > 0.5:
                self.gen.append('x')
            else:
                self.gen.append('1')
        else:
            val = random.choice(self.func_set[1] + self.func_set[2])
            self.gen.append(val)
            if val in self.func_set[2]:
                self.full(level + 1)
                self.full(level + 1)
            else:
                self.full(level + 1)
        
    def grow(self, level=0):
        """
        Function to generate a tree in a grow manner
        Every node may be a terminal or a function
        @return: None
        """
        if level == self.depth:
            if random.random() > 0.5:
                self.gen.append('x')
            else:
                self.gen.append('1')
        else:
            if random.random() > 0.3: # с вероятностью 0.7 продолжаем
                val = random.choice(self.func_set[1] + self.func_set[2])
                self.gen.append(val)
                if val in self.func_set[2]:
                    self.grow(level + 1)
                    self.grow(level + 1)
                else:
                    self.grow(level + 1)
            else: # с вероятностью 0.3 обрываем
                if random.random() > 0.5:
                    self.gen.append('x')
                else:
                    self.gen.append('1')
        
    def __evaluate(self, x, w, index=0):
        """
        Function to evaluateuate the current chromosome with a given x
        @param: x - function input
        @param: w - weights vector
        @index: current index in genotype
        @return: 
        """
        if self.gen[index] == 'x':
            return w[index] * x, index
        elif self.gen[index] == '1':
            return w[index], index
        elif self.gen[index] in self.func_set[2]:
            left, index_left = self.__evaluate(x, w, index + 1)
            right, index_right = self.__evaluate(x, w, index_left + 1)
            if self.gen[index] == '+':
                return w[index] * (left + right), index_right
            elif self.gen[index] == '-':
                return w[index] * (left - right), index_right
            elif self.gen[index] == '*':
                return w[index] * (left * right), index_right
            elif self.gen[index] == '/':
                return w[index] * (left / right), index_right
        else:
            left, index_left = self.__evaluate(x, w, index + 1)
            if self.gen[index] == 'sin':
                return w[index] * np.sin(left), index_left
            elif self.gen[index] == 'cos':
                return w[index] * np.cos(left), index_left
            elif self.gen[index] == 'exp':
                return w[index] * np.exp(left), index_left
            elif self.gen[index] == 'log':
                return w[index] * np.log(left), index_left
            elif self.gen[index] == 'ctg':
                return w[index] * 1/np.tan(left), index_left
            elif self.gen[index] == 'cth':
                return w[index] * 1/np.tanh(left), index_left

    def evaluate(self, x, w):
        """
        Function to evaluateuate the current genotype to a given input
        @return: the value of self.gen evaluateuated at the given input
        """
        return self.__evaluate(x, w)[0]

    def calculate_fitness(self, datasets=None, X=None, y=None):
        """
        Function to claculate the fitness of a chromosome
        @param X: inputs of the function we want to predict
        @param y: outputs of the function we want to predict
        @return: the chromosome's fitness (calculated based on MSE)
        """
        if datasets is not None:
            fitness = 0
            for name in datasets.keys():
                X = datasets[name]['sample_sizes']
                x_vals = X - min(X)
                x_vals = x_vals / max(x_vals)
                #y = datasets[name]['mean']
                y = datasets[name]['std']
                y_vals = y - min(y)
                y_vals = y_vals / max(y_vals)
                w_0 = np.zeros(len(self.gen))
                #print(w_0.size)
                problem = minimize(lambda w: np.array([(self.evaluate(x_vals[i], w) - y_vals[i])**2 for i in range(len(x_vals))]).mean(), w_0)
                w_opt = problem.x
                fitness += np.array([(self.evaluate(x_vals[i], w_opt) - y_vals[i])**2 for i in range(len(X))]).mean()
            fitness /= len(datasets)
            self.fitness = fitness
                
        elif X is not None and y is not None:
            w_0 = np.zeros(len(self.gen))
            #print(w_0.size)
            problem = minimize(lambda w: np.array([(self.evaluate(X[i], w) - y[i])**2 for i in range(len(X))]).mean(), w_0)
            w_opt = problem.x
            self.w_opt = w_opt
            fitness = np.array([(self.evaluate(X[i], w_opt) - y[i])**2 for i in range(len(X))]).mean()
            
        """
        diff = 0
        for i in range(len(X)):
            try:
                diff += (self.evaluate(X[i]) - y[i])**2
            except RuntimeWarning:
                self.gen = []
                if random.random() > 0.5:
                    self.grow()
                else:
                    self.full()
                self.calculate_fitness(X, y)
        """
    
        return fitness

    def __get_depth_aux(self, index=0):
        """
        Function to get the depth of a chromosome
        @return: chromosome's depth, last pos
        """
        elem = self.gen[index]
        
        if elem in self.func_set[2]:
            left, index_left = self.__get_depth_aux(index + 1)
            right, index_right = self.__get_depth_aux(index_left)
            return max(left, right) + 1, index_right
        elif elem in self.func_set[1]:
            left, index_left = self.__get_depth_aux(index + 1)
            return left + 1, index_left
        else:
            return 1, index + 1
        
    def get_depth(self):
        """
        Function to get the depth of a chromosome
        @return: - chromosome's depth
        """
        return self.__get_depth_aux()[0]
    
    def __reveal(self, index=0):
        if self.gen[index] == 'x':
            return f'w[{index}]*x', index
        elif self.gen[index] == '1':
            return f'w[{index}]', index
        elif self.gen[index] in self.func_set[2]:
            left, index_left = self.__reveal(index + 1)
            right, index_right = self.__reveal(index_left + 1)
            if self.gen[index] == '+':
                return f'w[{index}]*(' + left + '+' + right + ')', index_right
            elif self.gen[index] == '-':
                return f'w[{index}]*(' + left + '-' + right + ')', index_right
            elif self.gen[index] == '*':
                return f'w[{index}]*' + left + '*' + right, index_right
            elif self.gen[index] == '/':
                return f'w[{index}]*' + left + '/' + right, index_right
        else:
            left, index_left = self.__reveal(index + 1)
            if self.gen[index] == 'sin':
                return f'w[{index}]*sin({left})', index_left
            elif self.gen[index] == 'cos':
                return f'w[{index}]*cos({left})', index_left
            elif self.gen[index] == 'exp':
                return f'w[{index}]*exp({left})', index_left
            elif self.gen[index] == 'log':
                return f'w[{index}]*log({left})', index_left
            elif self.gen[index] == 'ctg':
                return f'w[{index}]*ctg({left})', index_left
            elif self.gen[index] == 'cth':
                return f'w[{index}]*cth({left})', index_left
        
    def __str__(self) -> str:
        return self.__reveal()[0]
    
    def __repr__(self) -> str:
        return self.__str__()