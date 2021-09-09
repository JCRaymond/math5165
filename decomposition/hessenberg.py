#!/usr/bin/env python3.9

import numpy as np


def givens_rot(n, i, a, j, b):
    G = np.eye(n)
    r = np.hypot(a, b)
    if b == 0 or r == 0:
        return G
    c = a / r
    s = b / r
    G[i, i] = c
    G[i, j] = s
    G[j, i] = -s
    G[j, j] = c
    return G


# Return two matrices G, X such that A = G * X * G.T where X is in Hessenberg
# form and G is an orthonormal matrix.
def hessenberg(A):
    n = A.shape[0]
    X = np.matrix(A)
    G = np.eye(n)
    for col in range(X.shape[1] - 2):
        for row in range(X.shape[0] - 1, col + 1, -1):
            G_i = givens_rot(n, row - 1, X[row - 1, col], row, X[row, col])
            G = G_i * G
            X = G_i * (X * G.T)
    return G, X


A = np.matrix([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 12, 11],
               [10, 9, 8, 7, 6], [5, 4, 3, 2, 1]])
A = np.matrix([[1, 2, 3, 4], [5, 6, 7, 8], [0, 7, 6, 5], [0, 0, 2, 1]])

print(hessenberg(A))
