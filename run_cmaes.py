'''
Example to run CMA-ES for SwarmRoBox2D.
SwarmRoBox2D-PythonScript (Author: Motoaki Hiraga)

Requires es.py from estool. 
https://github.com/hardmaru/estool

CMA-ES in estool is wrapping around pycma.
https://github.com/CMA-ES/pycma
'''

import ctypes
import os
import pickle
import time

import numpy as np

import cma
from es import CMAES


class Evaluator:
    def __init__(self, genomeLength):
        self.genomeLength = genomeLength
        self.generation = 0

    def setGeneration(self, generation):
        self.generation = generation

    def evaluateIndividual(self, individual):

        # libc = ctypes.CDLL("../SwarmRoBox2D/Debug/libSwarmRoBox2D.so")
        libc = ctypes.CDLL("../SwarmRoBox2D/Release/libSwarmRoBox2D.so")
        # libc = ctypes.CDLL("libSwarmRoBox2D/libSwarmRoBox2D.so")

        libc.evalFuncs.restype = ctypes.c_double

        seed = int(self.generation)
        func_index = 0

        genome = (ctypes.c_double * self.genomeLength)(*individual)

        fitness = libc.evalFuncs(genome, func_index, seed)

        return fitness


if __name__ == '__main__':

    lambdaCount = 200
    initStrategy = 0.2

    maxGenerationCount = 500

    inputNodeCount = 17
    hiddenNodeCount = 8
    outputNodeCount = 2

    genomeLength = (inputNodeCount * hiddenNodeCount +
                    hiddenNodeCount * hiddenNodeCount +
                    hiddenNodeCount * outputNodeCount +
                    inputNodeCount * outputNodeCount)

    randomSeed = 0  # int(time.time())
    np.random.seed(randomSeed)
    # TODO: CMA-ES in estool seems not to have a setting for a fixed seed value.
    #       Have to modify es.py for additional settings for pycma.

    data_dir = os.path.join(os.path.dirname(__file__),
                            'results_cmaes_{}'.format(randomSeed))
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

    # defines CMA-ES algorithm solver
    cmaes = CMAES(genomeLength, popsize=lambdaCount,
                  weight_decay=0.0, sigma_init=initStrategy)

    evaluator = Evaluator(genomeLength)

    log_stats = ['Gen', 'Mean', 'Std', 'Max', 'Min']

    with open('{}/log_stats.pkl'.format(data_dir), mode='wb') as out_pkl:
        pickle.dump(log_stats, out_pkl)

    for gen in range(maxGenerationCount):

        start_time = time.time()

        solutions = cmaes.ask()

        evaluator.setGeneration(gen)

        print('---')
        print('Generation {}'.format(gen))

        if multiProcessLib == 'mpi4py':
            fitnessValues = list(
                pool.map(evaluator.evaluateIndividual, solutions))
        else:
            if processCount > 1:
                fitnessValues = pool.map(
                    evaluator.evaluateIndividual, solutions)
            else:
                fitnessValues = []
                for ind in solutions:
                    fitnessValues += [evaluator.evaluateIndividual(ind)]

        cmaes.tell(fitnessValues)

        log_stats = [gen, np.mean(fitnessValues), np.std(fitnessValues),
                     np.max(fitnessValues), np.min(fitnessValues)]

        with open('{}/log_stats.pkl'.format(data_dir), mode='ab') as out_pkl:
            pickle.dump(log_stats, out_pkl)

        print('Mean: ' + str(np.mean(fitnessValues)) +
              '\tStd: ' + str(np.std(fitnessValues)) +
              '\tMax: ' + str(np.max(fitnessValues)) +
              '\tMin: ' + str(np.min(fitnessValues)))

        result = cmaes.result()

        elapsed_time = time.time() - start_time
        print('Elapsed time: {}[sec]'.format(elapsed_time))

    # Historically best individual and its fitness value.
    print('Best individual (fitness {}): '.format(result[1]))
    print(result[0])

    # Save the best individual.
    with open('{}/best_ind.pkl'.format(data_dir), mode='wb') as out_pkl:
        pickle.dump(result, out_pkl)
