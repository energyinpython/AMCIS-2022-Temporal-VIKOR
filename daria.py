import copy
import numpy as np
from normalizations import sum_normalization


class DARIA():
    def __init__(self):
        """
        Create the DARIA object
        """
        pass

    # gini coefficient
    def _gini(self, R):
        """
        Calculate variability values measured by the Gini coefficient in scores obtained by each evaluated option.
        
        Parameters
        -----------
            R : ndarray
                Matrix with preference values obtained with MCDA method (for example, TOPSIS)
                with `t` periods of time in rows and `m` alternatives in columns.
        
        Returns
        --------
            ndarray
                Vector with Gini coefficient values for each alternative.
        
        Examples
        ----------
        >>> daria = DARIA()
        >>> variability = daria._gini(matrix)
        """

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
        """
        Calculate variability values measured by the Entropy in scores obtained by each evaluated option.
        
        Parameters
        -----------
            R : ndarray
                Matrix with preference values obtained with MCDA method (for example, TOPSIS)
                with `t` periods of time in rows and `m` alternatives in columns.
        
        Returns
        --------
            ndarray
                Vector with Entropy values for each alternative.
        
        Examples
        ----------
        >>> daria = DARIA()
        >>> variability = daria._entropy(matrix)
        """

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
        """
        Calculate variability values measured by the Standard Deviation in scores obtained by each evaluated option.
        
        Parameters
        -----------
            R : ndarray
                Matrix with preference values obtained with MCDA method (for example, TOPSIS)
                with `t` periods of time in rows and `m` alternatives in columns.
        
        Returns
        --------
            ndarray
                Vector with Standard Deviation values for each alternative.
        
        Examples
        ----------
        >>> daria = DARIA()
        >>> variability = daria._std(matrix)
        """

        return np.sqrt((np.sum(np.square(R - np.mean(R, axis = 0)), axis = 0))/R.shape[0])


    # statistical variance
    def _stat_var(self, X):
        """
        Calculate variability values measured by the Statistical Variance in scores obtained by each evaluated option.
        
        Parameters
        -----------
            R : ndarray
                Matrix with preference values obtained with MCDA method (for example, TOPSIS)
                with `t` periods of time in rows and `m` alternatives in columns.
        
        Returns
        --------
            ndarray
                Vector with Statistical Variance values for each alternative.
        
        Examples
        ----------
        >>> daria = DARIA()
        >>> variability = daria._stat_var(matrix)
        """

        v = np.mean(np.square(X - np.mean(X, axis = 0)), axis = 0)
        return v


    # for MCDA methods type = 1: descending order: higher is better, type -1: opposite
    def _direction(self, R, type):
        """
        Determine the direction of the variability of alternatives scores obtained in the following 
        periods of time.
        
        Parameters
        ------------
            R : ndarray
                Matrix with preference values obtained with MCDA method (for example, TOPSIS)
                with `t` periods of time in rows and `m` alternatives in columns.
            type : int
                The variable represents the ordering of alternatives by the MCDA method. It can be equal to
                1 or -1. 1 means that the MCDA method sorts options in descending order
                according to preference values (for example, the TOPSIS method). -1 means that 
                the MCDA method sorts options in ascending order according to preference values 
                (for example, the VIKOR method). 
        
        Returns
        --------
            direction_list : list
                List with strings representing the direction of variability in the form of the
                arrow up for improvement, arrow down for worsening, and = for stability.
                It is useful for results presentation.
            dir_class : ndarray
                Vector with numerical values representing the direction of variability. 1 represents
                increasing preference values, and -1 means decreasing preference values.
                It is used to calculate final aggregated preference values using DARIA method in
                next stage of DARIA method.
        
        Examples
        ---------
        >>> daria = DARIA()
        >>> dir_list, dir_class = daria._direction(matrix, type)
        """

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
        """
        Calculate final aggregated preference values of alternatives of DARIA method.
        Obtained preference values can be sorted according to chosen MCDA method rule to generate
        ranking of alternatives.
        
        Parameters
        -----------
            S : ndarray
                Vector with preference values of alternatives from the most recent year analyzed
                obtained by chosen MCDA method.
            G : ndarray
                Vector with variability values of alternatives preferences obtained in investigated
                periods.
            dir : ndarray
                Vector with numerical values of the direction of variability in values of alternatives 
                preferences obtained in investigated periods. 1 represents increasing in following
                preference values, and -1 means decreasing in following preference values.
        
        Returns
        --------
            ndarray
                Final aggregated preference values of alternatives considering variability in
                preference values obtained in the following periods.
        
        Examples
        ----------
        >>> final_S = daria._update_efficiency(S, G, dir)
        >>> rank = rank_preferences(final_S, reverse = True)
        """
        
        return S + G * dir