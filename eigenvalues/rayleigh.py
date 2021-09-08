#!/usr/bin/env python3.9

import numpy as np
from sys import stderr


def normalize(vec):
    return vec / np.linalg.norm(vec)


def col(vec):
    return np.atleast_2d(vec).T


def rayleigh_e_val_vec(A, epsilon=1e-6):
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        return None

    n = A.shape[0]
    x = normalize(col(np.random.rand(A.shape[1])))
    # No need to divide by x.T * x because x is normalized, so x.T*x = 1
    rayleigh = (x.T * (A * x)).item()

    dist = 1 + epsilon

    iters = 0
    while dist > epsilon:
        x_old = x

        # Perform two iterations to handle negative eigenvalues
        try:
            A_s = np.linalg.inv(A - rayleigh * np.eye(n))
        except np.linalg.LinAlgError:
            break
        y = normalize(A_s * x)
        rayleigh = (y.T * (A * y)).item()

        try:
            A_s = np.linalg.inv(A - rayleigh * np.eye(n))
        except np.linalg.LinAlgError:
            x = y
            break
        x = normalize(A_s * y)
        rayleigh = (x.T * (A * x)).item()

        iters += 2

        dist = np.linalg.norm(x - x_old, ord=np.inf)

    print(f'Num Iters: {iters}', file=stderr)

    return rayleigh, x


if __name__ == "__main__":

    A = np.matrix([[-1, 0, 0], [0, 2, 0], [0, 0, -3]])
    val, vec = rayleigh_e_val_vec(A)

    print(val, vec)
