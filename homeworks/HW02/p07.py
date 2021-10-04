#!/usr/bin/env python3.9

import numpy as np
import numpy.linalg as la


def col(lst):
    return np.atleast_2d(np.array(lst)).T


# For a matrix A, return matrices Q and H such that Q is orthogonal
# and H is in Hessenberg form such that A = QHQ.T
def Hessenberg(A):
    n = A.shape[0]
    Q = np.eye(n)
    H = np.matrix(A)

    for i in range(n - 2):
        h = np.matrix(H[i + 1:, i])
        h_norm = la.norm(h)
        if h[0] >= 0:
            h[0] += h_norm
        else:
            h[0] -= h_norm
        h /= la.norm(h)

        Q[:, i + 1:] -= (Q[:, i + 1:] @ h) @ (2 * h.T)
        H[i + 1:, i:] -= (2 * h) @ (h.T @ H[i + 1:, i:])
        H[i:, i + 1:] -= (H[i:, i + 1:] @ h) @ (2 * h.T)

    return Q, H


def solve():
    A = np.matrix([[1., 2, 3], [2, 4, 5], [3, 5, 7]])

    Q, H = Hessenberg(A)

    print("Original Matrix:")
    print(A)

    print("Q:")
    print(Q)
    print("H:")
    print(H)
    print("QHQ.T:")
    print(Q @ H @ Q.T)


if __name__ == '__main__':
    solve()
