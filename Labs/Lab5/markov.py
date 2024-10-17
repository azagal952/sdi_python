import numpy as np
import math

def markov(rho, transition_matrix, nmax, rng):
    """Computes a Markov chain, with initial distribution rho and transition matrix A.

    Args:
        rho (np.ndarray): the initial distribution (shape N*1, in the unit simplex)
        transition_matrix: the transition matrix (shape N*N, must be stochastic)
        nmax: maximum number of iterations
        rng: a random number generator

    Returns:
        np.ndarray: the trajectory of the Markov chain, of length nmax + 1.
    """
    # The transition matrix must be square matrix.
    assert transition_matrix.shape[0] == transition_matrix.shape[1]
    # The size of transition matrix must be equal to the length of rho
    assert transition_matrix.shape[0] == len(rho)
    # The sum of tho coefficients must be equal to 1
    assert math.isclose(np.sum(rho), 1, rel_tol=1e-7)

    N = len(rho)
    X = np.zeros(nmax + 1, dtype=int)
    E = np.arange(1, N + 1)

    X[0] = rng.choice(E, p=rho)
    for i in range(nmax):
        X[i + 1] = rng.choice(E, p=transition_matrix[X[i] - 1, :])
    return X

