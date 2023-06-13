'''
Example to run self-adaptive evolution strategies with a set of strategy parameters for SwarmRoBox2D.
SwarmRoBox2D-PythonScript (Author: Motoaki Hiraga)
'''

import copy
import ctypes
import math
import os
import pickle
import random
import time

import numpy as np


class Individual:

    def __init__(self, genomeLength, minWeight, maxWeight, minStrategy, maxStrategy, initStrategy):

        self.genomeLength = genomeLength

        self.minWeight = minWeight
        self.maxWeight = maxWeight

        self.minStrategy = minStrategy
        self.maxStrategy = maxStrategy

        self.genome = [random.uniform(self.minWeight, self.maxWeight)
                       for i in range(genomeLength)]
        self.strategies = [initStrategy for i in range(genomeLength)]

        self.fitness = 0.0
        self.generation = 0


def mutate(ind, c=1.0):
    N = random.gauss(0.0, 1.0)
    tau_ = c / math.sqrt(2.0 * ind.genomeLength)
    tau = c / math.sqrt(2.0 * math.sqrt(ind.genomeLength))
    new_s = []
    new_g = []
    for s, g in zip(ind.strategies, ind.genome):
        s *= math.exp(tau_ * N + tau * random.gauss(0.0, 1.0))
        s = np.clip(s, ind.minStrategy, ind.maxStrategy)
        g += s * random.gauss(0.0, 1.0)
        g = np.clip(g, ind.minWeight, ind.maxWeight)
        # g += random.gauss(0.0, 1.0)
        # g = np.clip(g, ind.minWeight, ind.maxWeight)
        new_s += [s]
        new_g += [g]
    ind.genome = new_g
    ind.strategies = new_s
    ind.generation += 1


def evaluateIndividual(individual):

    # libc = ctypes.CDLL("../SwarmRoBox2D/Debug/libSwarmRoBox2D.so")
    libc = ctypes.CDLL("../SwarmRoBox2D/Release/libSwarmRoBox2D.so")
    # libc = ctypes.CDLL("libSwarmRoBox2D/libSwarmRoBox2D.so")

    libc.evalFuncs.restype = ctypes.c_double

    seed = int(individual.generation)
    func_index = 0

    genome = (ctypes.c_double * len(individual.genome))(*individual.genome)

    fitness = libc.evalFuncs(genome, func_index, seed)

    return fitness


if __name__ == '__main__':

    muCount = 30
    lambdaCount = 200

    maxGenerationCount = 500
    isMaximizingFitness = True

    minWeight = -1.0
    maxWeight = 1.0

    minStrategy = 0.001
    maxStrategy = 0.5

    initStrategy = 0.2

    inputNodeCount = 17
    hiddenNodeCount = 8
    outputNodeCount = 2

    genomeLength = (inputNodeCount * hiddenNodeCount +
                    hiddenNodeCount * hiddenNodeCount +
                    hiddenNodeCount * outputNodeCount +
                    inputNodeCount * outputNodeCount)

    randomSeed = 0  # int(time.time())
    random.seed(randomSeed)

    data_dir = os.path.join(os.path.dirname(__file__),
                            'results_es_{}'.format(randomSeed))
    os.makedirs(data_dir, exist_ok=True)

    multiProcessLib = 'multiprocessing'  # 'mpi4py' or 'multiprocessing'

    if multiProcessLib == 'mpi4py':
        from mpi4py.futures import MPIPoolExecutor
        pool = MPIPoolExecutor()

    elif multiProcessLib == 'multiprocessing':
        import multiprocessing
        processCount = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=processCount)
    else:
        processCount = 1

    pop = [Individual(genomeLength, minWeight, maxWeight, minStrategy, maxStrategy, initStrategy)
           for i in range(lambdaCount)]

    log_stats = ['Gen', 'Mean', 'Std', 'Max', 'Min']

    with open('{}/log_stats.pkl'.format(data_dir), mode='wb') as out_pkl:
        pickle.dump(log_stats, out_pkl)

    for gen in range(maxGenerationCount):

        start_time = time.time()

        print('---')
        print('Generation {}'.format(gen))

        if multiProcessLib == 'mpi4py':
            fitnessValues = list(pool.map(evaluateIndividual, pop))
        else:
            if processCount > 1:
                fitnessValues = pool.map(evaluateIndividual, pop)
            else:
                fitnessValues = []
                for ind in pop:
                    fitnessValues += [evaluateIndividual(ind)]

        for ind, fit in zip(pop, fitnessValues):
            ind.fitness = fit

        log_stats = [gen, np.mean(fitnessValues), np.std(fitnessValues),
                     np.max(fitnessValues), np.min(fitnessValues)]

        with open('{}/log_stats.pkl'.format(data_dir), mode='ab') as out_pkl:
            pickle.dump(log_stats, out_pkl)

        print('Mean: ' + str(np.mean(fitnessValues)) +
              '\tStd: ' + str(np.std(fitnessValues)) +
              '\tMax: ' + str(np.max(fitnessValues)) +
              '\tMin: ' + str(np.min(fitnessValues)))

        pop.sort(key=lambda ind: ind.fitness, reverse=isMaximizingFitness)

        print('Best individual (fitness {}): '.format(pop[0].fitness))
        print(pop[0].genome)

        # Save the best individual.
        with open('{}/data_ind_gen{:0>4}.pkl'.format(data_dir, gen), mode='wb') as out_pkl:
            pickle.dump(pop[0], out_pkl)

        # Select lambda individuals from top mu individuals
        new_pop = random.choices(pop[0:muCount], k=lambdaCount)
        pop = [copy.deepcopy(i) for i in new_pop]

        for i in pop:
            mutate(i)

        elapsed_time = time.time() - start_time
        print('Elapsed time: {}[sec]'.format(elapsed_time))
