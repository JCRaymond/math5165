#!/usr/bin/env python3.9

import numpy as np
import numpy.linalg as la


def col(lst):
    return np.atleast_2d(np.array(lst)).T


def givens(a, b, epsilon=1e-10, G=None):
    if G is None:
        G = np.matrix((2, 2))
    r = np.hypot(a, b)
    if np.abs(b) < epsilon or np.abs(r) < epsilon:
        G[0, 0] = 1
        G[0, 1] = 0
        G[1, 0] = 0
        G[1, 1] = 1
        return G
    c = a / r
    s = b / r
    G[0, 0] = c
    G[0, 1] = s
    G[1, 0] = -s
    G[1, 1] = c
    return G


# For a matrix A = QR, where Q is an orthogonal matrix and R is an upper
# triangular matrix, return a QR decomposition of A + uv.T
#
# Time Complexity: O(n^2)
# Space Complexity: O(n) (not counting output)
def shift_QR(Q, R, u, v):
    Q_prime = np.matrix(Q)  # ops = n^2 -=- copy Q
    R_prime = np.matrix(R)  # ops = n^2 -=- copy R

    n = Q.shape[0]  # ops = O(1)
    w = Q.T @ u  # ops = n^2 -=- compute w

    b = w[n - 1, 0]  # ops = O(1)
    G = np.eye(2)  # ops = O(1)
    for i in range(n - 2, -1, -1):  # iterate i = n-2, n-1, ..., 1, 0
        a = w[i]  # ops = O(1)
        G = givens(a, b,
                   G=G)  # ops = O(1) -=- creating 2x2 matrix is constant time

        # ops = 6 * (n-i) -=- apply Givens rotation: 6 flops/column, n-i columns
        R_prime[i:i + 2, i:] = G @ R_prime[i:i + 2, i:]

        # ops = 6 * n -=- apply Givens rotation: 6 flops/column, n columns
        Q_prime[:, i:i + 2] = Q_prime[:, i:i + 2] @ G.T

        b = np.hypot(a, b)  # ops = O(1)
    R_prime[
        0, :] += b * v.T  # ops = 2 * n -=- combine u,v into R_prime (b = ||u||)

    for i in range(n - 1):  # iterate i = 0, 1, ..., n-2, n-1
        a = R_prime[i, i]  # ops = O(1)
        b = R_prime[i + 1, i]  # ops = O(1)
        G = givens(a, b, G=G)  # ops = O(1)

        R_prime[i:i + 2, i:] = G @ R_prime[i:i + 2, i:]  # ops = 6 * (n-i)
        R_prime[i + 1, i] = 0  # ops = O(1) -=- 0 entry that should be 0

        Q_prime[:, i:i + 2] = Q_prime[:, i:i + 2] @ G.T  # ops = 6 * n

    return Q_prime, R_prime


# For a matrix A, return a QR decomposition of A
def QR(A):
    n = A.shape[0]  # ops = O(1)
    Q = np.eye(n)  # ops = n^2
    R = np.matrix(A)  # ops = n^2

    for i in range(n - 1):  # iterate i = 0, 1, ..., n-2
        h = np.matrix(R[i:, i])  # ops = n-i -=- copy bottom of ith column of R
        h_norm = la.norm(h)  # ops = 2n

        # ops = O(1)
        if h[0] >= 0:
            h[0] += h_norm
        else:
            h[0] -= h_norm

        h /= la.norm(h)  # ops = 3n

        Q[:, i:] -= (Q[:, i:] @ h) @ (2 * h.T)  # ops = 3*n*(n-i) + (n-i)
        R[i:, i:] -= (2 * h) @ (h.T @ R[i:, i:])  # ops = 3*(n-i)^2 + (n-i)

    return Q, R


def solve():
    A = np.matrix([[1., 2, 3, 4], [2, 4, 6, 7], [3, 6, 8, 10], [4, 7, 10, 13]])
    u = col([1., 3, 5, 7])
    v = col([2., 4, 6, 8])

    print('=== Original Matrix ===')
    print('A:')
    print(A)
    print()

    print('=== Shifting vectors ===')
    print('u:')
    print(u)
    print('v:')
    print(v)
    print()

    A_p = A + u @ v.T
    print('=== Shifted Matrix ===')
    print('A + uv.T:')
    print(A_p)
    print()

    # Get a QR decomposition for A
    Q, R = QR(A)
    print('=== QR decomposition of A ===')
    print('Q:')
    print(Q)
    print('R:')
    print(R)
    print()

    # Shift the decomposition by uv.T
    Q_p, R_p = shift_QR(Q, R, u, v)

    print('=== Shifted QR decomposition of A + uv.T ===')
    print('Shifted Q:')
    print(Q_p)
    print('Shifted Q @ Shift Q.T (should be close to I):')
    print(Q_p @ Q_p.T)
    print('Shifted R:')
    print(R_p)
    print('Shifted Q @ Shifted R:')
    print(Q_p @ R_p)
    print()


if __name__ == '__main__':
    solve()
