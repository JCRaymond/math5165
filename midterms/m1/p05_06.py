#!/usr/bin/env python3.9

import numpy as np
import numpy.linalg as la


def col(lst):
    return np.atleast_2d(np.array(lst)).T


# Performs an in-place application of the Householder transformation
# corresponding to v to the left of A (SA).
def householder_left(v, A):
    A[:, :] -= 2 * v @ (v.T @ A)  # O(n^2) + O(n^2) + O(n^2)


# Performs an in-place application of the transpose of the Householder
# transformation corresponding to v to the right of A (AS.T)
def householder_right_T(A, v):
    A[:, :] -= 2 * ((A @ v.T) @ v)


# For a matrix A = QR, where Q is an orthogonal matrix and R is an upper
# triangular matrix, return a QR decomposition of A + uv.T
def shift_QR(Q, R, u, v):
    h = Q.T @ u  # O(n^2)

    h_norm = la.norm(h)  # O(n)

    # O(1)
    if h[0] >= 0:
        h[0] += h_norm
    else:
        h[0] -= h_norm

    h /= la.norm(h)  # O(n)

    R_prime = R - (2 * h) @ (h.T @ R)  # O(n) + O(n^2) + O(n^2)
    R_prime[0, :] += h_norm * v.T  # O(n)

    Q_prime = Q - (Q @ h) @ (2 * h.T)  # O(n^2) + O(n) + O(n^2)

    return Q_prime, R_prime


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
    A = np.matrix([[1., 3, 8], [1, 2, 6], [0, 1, 2]])
    u = col([1., 2, 3])
    v = col([4., 5, 6])

    Q, R = QR(A)
    print(A)
    print(Q @ R)

    A_1 = A + (u @ v.T)
    Q_1, R_1 = QR(A_1)
    print(A_1)
    print(Q_1 @ R_1)

    Q_2, R_2 = shift_QR(Q, R, u, v)
    print(Q_2 @ R_2)

    print()

    print(Q_1)
    print(Q_2)

    print()

    print(R_1)
    print(R_2)


if __name__ == '__main__':
    solve()
