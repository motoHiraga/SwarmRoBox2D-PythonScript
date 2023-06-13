'''
Example to decode ES individuals for SwarmRoBox2D.
SwarmRoBox2D-PythonScript (Author: Motoaki Hiraga)
'''

import pickle
import numpy as np


class Individual:

    def __init__(self, genomeLength, minWeight, maxWeight, minStrategy, maxStrategy, initStrategy):

        self.genomeLength = genomeLength

        self.minWeight = minWeight
        self.maxWeight = maxWeight

        self.minStrategy = minStrategy
        self.maxStrategy = maxStrategy

        self.genome = []
        self.strategies = []

        self.fitness = 0.0
        self.generation = 0


if __name__ == '__main__':
    path = 'results_es_0'
    gen = '499'

    with open('{}/data_ind_gen{:0>4}.pkl'.format(path, gen), 'rb') as pkl:
        ind = pickle.load(pkl)

    # Use the generation number as the random seed of the simulation.
    seed = ind.generation

    print('---------------------------------')
    print('Fitness value: {}'.format(ind.fitness))
    print('---------------------------------\n')

    print('float genotype[genotypeLength] = {')
    print(*ind.genome, sep=', ')
    print('};\n')

    print('int32 seed = {};\n'.format(seed))
