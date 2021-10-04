#!/usr/bin/env python3.9

import numpy as np
import numpy.linalg as la


def col(lst):
    return np.atleast_2d(np.array(lst)).T


# For a matrix A, return a QR decomposition of A
def QR(A):
    n = A.shape[0]
    Q = np.eye(n)
    R = np.matrix(A)

    for i in range(n - 1):
        h = np.matrix(R[i:, i])
        h_norm = la.norm(h)
        if h[0] >= 0:
            h[0] += h_norm
        else:
            h[0] -= h_norm
        h /= la.norm(h)

        Q[:, i:] -= (Q[:, i:] @ h) @ (2 * h.T)
        R[i:, i:] -= (2 * h) @ (h.T @ R[i:, i:])

    return Q, R


def solve():
    A = np.matrix([[1., 2, 3], [2, 4, 5], [3, 5, 7]])

    Q, R = QR(A)

    print("Original Matrix:")
    print(A)

    print("Q:")
    print(Q)
    print("R:")
    print(R)
    print("QR:")
    print(Q @ R)


if __name__ == '__main__':
    solve()
