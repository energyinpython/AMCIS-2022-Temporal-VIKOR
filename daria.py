import copy
import numpy as np
from normalizations import sum_normalization


class DARIA():
    def __init__(self):
        pass

    # gini coefficient
    def _gini(self, R):
        t, m = np.shape(R)
        G = np.zeros(m)
        # iteration over alternatives i=1, 2, ..., m
        for i in range(0, m):
            # iteration over periods p=1, 2, ..., t
            Yi = np.zeros(t)
            if np.mean(R[:, i]) != 0:
                for p in range(0, t):
                    for k in range(0, t):
                        Yi[p] += np.abs(R[p, i] - R[k, i]) / (2 * t**2 * (np.sum(R[:, i]) / t))
            else:
                for p in range(0, t):
                    for k in range(0, t):
                        Yi[p] += np.abs(R[p, i] - R[k, i]) / (t**2 - t)
            G[i] = np.sum(Yi)
        return G


    # entropy
    def _entropy(self, R):
        # normalization for profit criteria
        criteria_type = np.ones(np.shape(R)[1])
        pij = sum_normalization(R, criteria_type)
        m, n = np.shape(pij)

        H = np.zeros((m, n))
        for j in range(n):
            for i in range(m):
                if pij[i, j] != 0:
                    H[i, j] = pij[i, j] * np.log(pij[i, j])

        h = (-1 * np.sum(H, axis = 0)) / (np.log(m))
        return 1 - h


    # standard deviation
    def _std(self, R):
        return np.sqrt((np.sum(np.square(R - np.mean(R, axis = 0)), axis = 0))/R.shape[0])


    # statistical variance
    def _stat_var(self, X):
        v = np.mean(np.square(X - np.mean(X, axis = 0)), axis = 0)
        return v


    # for MCDA methods type = 1: descending order: higher is better, type -1: opposite
    def _direction(self, R, type):
        t, m = np.shape(R)
        direction_list = []
        dir_class = np.zeros(m)
        # iteration over alternatives i=1, 2, ..., m
        for i in range(m):
            thresh = 0
            # iteration over periods p=1, 2, ..., t
            for p in range(1, t):
                thresh += R[p, i] - R[p - 1, i]
            # classification based on thresh
            dir_class[i] = np.sign(thresh)
        direction_array = copy.deepcopy(dir_class)
        direction_array = direction_array * type
        for i in range(len(direction_array)):
            if direction_array[i] == 1:
                direction_list.append(r'$\uparrow$')
            elif direction_array[i] == -1:
                direction_list.append(r'$\downarrow$')
            elif direction_array[i] == 0:
                direction_list.append(r'$=$')
        return direction_list, dir_class


    def _update_efficiency(self, S, G, dir):
        return S + G * dir