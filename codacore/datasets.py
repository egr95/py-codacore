
import numpy as np

def simulate_hts(n, p, logratio='simple', random_state=None, zero_replace=True):
    """
    This function simulates a set of (x, y) pairs.
    The covariates x are compositional, meaning they only
    carry relative information.
    The response y is a binary indicator.
    The rule linking x and y can be a balance or an amalgamation.

    :param n:
    :param p:
    :param logratio:
    :param random_state:
    :return:
    """

    if random_state is not None:
        np.random.seed(random_state)

    # Simulate independent variables
    alpha0 = np.ones(p) / np.log(p)
    alpha = np.random.dirichlet(alpha0)
    alpha = -np.sort(-alpha)
    P = np.random.dirichlet(alpha, size=n)
    row_sums = np.random.poisson(lam=10 * p, size=n)
    X = np.ndarray([n, p])

    for i in range(n):
        x = np.random.multinomial(n=row_sums[i], pvals=P[i, :], size=1)
        X[i, :] = x

    # Simulate dependent variable
    if logratio == 'simple':
        if p < 2:
            raise ValueError("Input dimension must be >= 2")
        eta = np.log(P[:, 0]) - np.log(P[:, 1])

    elif logratio == 'balance':
        if p < 10:
            raise ValueError("Input dimension must be >= 10")
        eta = np.mean(np.log(P[:, [3, 5]]), axis=1) - np.mean(np.log(P[:, [4]]), axis=1)

    elif logratio == 'amalgamation':
        if p < 20:
            raise ValueError("Input dimension must be >= 20")

        eta = np.log(np.sum(P[:, [0, 1, 5, 6, 14]], axis=1)) - \
              np.log(np.sum(P[:, [2, 7, 15, 16]], axis=1))

    else:
        raise ValueError("Variable logratio incorrectly specified.")

    out_prob = 1 / (1 + np.exp(-(eta - np.mean(eta)))) * 1.0
    y = np.random.binomial(1, out_prob, n)

    if zero_replace:
        X = X + 1

    return X, y
