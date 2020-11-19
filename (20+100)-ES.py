import numpy as np
import math
import random


# class to store the solutions and their fitness levels as objects
class Solution:
    def __init__(self, x, fit):
        self.x = x
        self.fit = fit


# methods for the GA operators
def crossover(P):  # arithmetic mean
    sum = 0
    for i in range(p):
        sum += random.choice(P).x
    return sum / p


def mutation(x):  # gaussian mutation
    return x + sigma * np.random.randn(N)


def fitness_sphere(x):  # sphere function
    return np.dot(x.T, x)


def fitness_rastrigin(x):  # rastrigin function
    fit = 10. * len(x)
    for i in range(len(x)):
        fit += x[i] ** 2. - 10. * math.cos(2. * math.pi * x[i])
    return fit


def fitness_rosenbrock(x):  # rosenbrock function
    fit = 0
    for i in range(len(x) - 1):
        fit += 100. * (x[i-1] - x[i] ** 2.) ** 2. + (1. - x[i]) ** 2.
    return fit


def selection(population):  # plus selection
    return sorted(population, key=lambda solution: solution.fit)[:mu]


# other methods
def rechenberg(sigma, population, previous_fitness):

    success_rate = sum(solution.fit < previous_fitness for solution in population) / len(population)
    if success_rate < (1/5):
        indicator = 0
    elif success_rate > (1/5):
        indicator = 1
    else:
        return sigma
    sigma *= math.exp(indicator - (1/5))
    return sigma


def restart(pop, noise_strength):
    for solution in population:
        solution.x += noise_strength * np.random.randn(N)
    return pop


# variables
N = 10
generations_number = 100

# given mu, lambda, p
mu = 20  # number of parent solutions
lambda_ = 100  # number of offspring solutions
p = 2  # number of parents for crossover

# initialize P (parental population)
parents = []
for i in range(mu):
    x = np.random.randn(N) # randomly start in an interval
    fitness = fitness_sphere(x) # todo extension: restart strategy
    obj = Solution(x, fitness)
    parents.append(obj)
generation = 1
population = parents
best_solution = sorted(parents, key=lambda x: x.fit)[0]
print("generation:", generation, "fitness:", best_solution.fit)

# initialize sigma
sigma = 0.1

# repeat until termination condition (100 generations simulated)
while generation < generations_number:
    # for 100 offspring solutions: create by crossover and mutation, compute fitness
    offspring = []
    for i in range(lambda_):
        x = crossover(parents)  # crossover
        x = mutation(x)  # mutation
        fitness = fitness_sphere(x)  # fitness
        obj = Solution(x, fitness)
        offspring.append(obj)
    generation += 1
    average_fitness = sum(solution.fit for solution in population) / len(population)
    population = parents + offspring

    #  rechenberg mutation rate control
    sigma = rechenberg(sigma, population, average_fitness)

    #  gaussian restart
    new_best_solution = sorted(population, key=lambda x: x.fit)[0]
    if new_best_solution.fit == best_solution.fit:
        population = restart(population, 10)

    # end this iteration by selecting the mu best solutions from the parental population and offspring
    parents = selection(population)  # selection

    best_solution = sorted(parents, key=lambda x: x.fit)[0]
    print("generation:", generation, "fitness:", best_solution.fit, "sigma", sigma)
