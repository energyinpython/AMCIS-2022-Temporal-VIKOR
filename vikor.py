import numpy as np
from mcdm_method import MCDM_method


class VIKOR(MCDM_method):
    def __init__(self, normalization_method = None, v = 0.5):
        self.v = v
        self.normalization_method = normalization_method

    def __call__(self, matrix, weights, types):
        VIKOR._verify_input_data(matrix, weights, types)
        return VIKOR._vikor(matrix, weights, types, self.normalization_method, self.v)

    @staticmethod
    def _vikor(matrix, weights, types, normalization_method, v):
        if normalization_method == None:
            ind_profit = np.where(types == 1)[0]
            ind_cost = np.where(types == -1)[0]

            maximums_matrix = np.amax(matrix, axis = 0)
            minimums_matrix = np.amin(matrix, axis = 0)

            fstar = np.zeros(matrix.shape[1])
            fmin = np.zeros(matrix.shape[1])

            fstar[ind_profit] = maximums_matrix[ind_profit]
            fstar[ind_cost] = minimums_matrix[ind_cost]
            fmin[ind_profit] = minimums_matrix[ind_profit]
            fmin[ind_cost] = maximums_matrix[ind_cost]

            weighted_matrix = weights * ((fstar - matrix) / (fstar - fmin))
        else:
            norm_matrix = normalization_method(matrix, types)
            fstar = np.amax(norm_matrix, axis = 0)
            fmin = np.amin(norm_matrix, axis = 0)
            weighted_matrix = weights * ((fstar - norm_matrix) / (fstar - fmin))

        
        S = np.sum(weighted_matrix, axis = 1)
        R = np.amax(weighted_matrix, axis = 1)
        Sstar = np.min(S)
        Smin = np.max(S)
        Rstar = np.min(R)
        Rmin = np.max(R)
        Q = v * (S - Sstar) / (Smin - Sstar) + (1 - v) * (R - Rstar) / (Rmin - Rstar)
        return Q
