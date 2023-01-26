import numpy as np
from numpy import ma

def bootstrap_mat(mat, b=100):
    """We bootstrap the degenerate U statistic assuming identical p, q.
    We can't use a usual bootstrap (wo -1/N) as dicussed in Arcones & Gine 1992:
    We want to perform the bootstrap on the first non-zero term in the Hoeffding decomposition,
    which in this case is the same U-stat with h(x, y) made degenerate. This sum will stay
    O(1/N) while the actual U-stat is O(1/sqrt(N))."""
    N = len(mat)
    ws = np.random.multinomial(N, np.ones(N)/N, b)
    mats = np.einsum('ij,bi,bj->bij', mat, (ws-1)/N, (ws-1)/N)
    return [ma.masked_array(mat_i, np.eye(len(mat))).sum() for mat_i in mats]

def lin_reg_error(k_yx, k_xx, y_train, y_test, regularize=0.001):
    """ Calculate the error from kernel regression """
    reg_design_mat = k_xx + regularize * np.eye(len(k_xx))
    predictions = np.dot(k_yx, np.linalg.solve(reg_design_mat, y_train))
    error = np.average((predictions - y_test)**2)
    return error
