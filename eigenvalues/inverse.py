#!/usr/bin/env python3.9

import numpy as np
from sys import stderr


def normalize(vec):
    return vec / np.linalg.norm(vec)


def col(vec):
    return np.atleast_2d(vec).T


def inverse_eigenvec(A, epsilon=1e-6):
    if len(A.shape) != 2 or A.shape[0] != A.shape[1] or np.linalg.det(A) == 0:
        return None

    n = A.shape[0]
    x = normalize(col(np.random.rand(n)))

    A_inv = np.linalg.inv(A)

    dist = 1 + epsilon

    iters = 0
    while dist > epsilon:
        x_old = x

        # Perform two iterations to handle negative eigenvalues
        x = normalize(A_inv * (A_inv * x))
        iters += 2

        dist = np.linalg.norm(x - x_old, ord=np.inf)

    print(f'Num Iters: {iters}', file=stderr)

    return x


def inverse_e_val_vec(A, epsilon=1e-6):
    e_vec = inverse_eigenvec(A, epsilon)
    # Rayleigh quotient of an eigenvector is its eigenvalue
    e_val = (e_vec.T * (A * e_vec)).item()
    return e_val, e_vec


if __name__ == "__main__":

    A = np.matrix([[-1, 0, 0], [0, 2, 0], [0, 0, -3]])
    val, vec = inverse_e_val_vec(A)

    print(val, vec)
