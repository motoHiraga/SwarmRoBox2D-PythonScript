'''
Example to decode CMA-ES individuals for SwarmRoBox2D.
SwarmRoBox2D-PythonScript (Author: Motoaki Hiraga)
'''

import pickle
import numpy as np


if __name__ == '__main__':
    path = 'results_cmaes_0'

    with open('{}/best_ind.pkl'.format(path), 'rb') as pkl:
        ind = pickle.load(pkl)

    print('Fitness value: {}'.format(ind[1]))
    print(list(ind[0]))
