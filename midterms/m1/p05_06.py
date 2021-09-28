#!/usr/bin/env python3.9

import numpy as np
import numpy.linalg as la


def col(lst):
    return np.atleast_2d(np.array(lst)).T


##################
# SOLUTION TO #5 #
##################


# For a matrix A = QHQ.T, where Q is an orthogonal matrix and H is a
# Hessenberg matrix, return a Hessenberg decomposition of A + uv.T
def shift_Hessenberg(Q, H, u, v): 
    h = Q.T @ u  # O(n^2)
    h[0] = 0

    h_norm = la.norm(h)  # O(n)

    # O(1)
    if h[1] >= 0:
        h[1] += h_norm
    else:
        h[1] -= h_norm

    h /= la.norm(h)  # O(n)

    H_prime = H - (2 * h) @ (h.T @ R)  # O(n) + O(n^2) + O(n^2)
    R_prime[0, :] += h_norm * (v.T @ Q)  # O(n)

    Q_prime = Q - (Q @ h) @ (2 * h.T)  # O(n^2) + O(n) + O(n^2)

    return Q_prime, R_prime


# For a matrix A, return matrices Q and H such that Q is orthogonal
# and H is in Hessenberg form such that A = QHQ.T
def Hessenberg(A):
    n = A.shape[0]
    Q = np.eye(n)
    H = np.matrix(A)

    for i in range(n-2):
        h = np.matrix(H[i+1:, i])
        h_norm = la.norm(h)
        if h[0] >= 0:
            h[0] += h_norm
        else:
            h[0] -= h_norm
        h /= la.norm(h)

        Q[:, i+1:] -= (Q[:, i+1:] @ h) @ (2 * h.T)
        H[i+1:, i:] -= (2 * h) @ (h.T @ H[i+1:, i:])
        H[i:, i+1:] -= (H[i:, i+1:] @ h) @ (2 * h.T)
    
    return Q, H


##################
# SOLUTION TO #6 #
##################


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

    Q, H = Hessenberg(A)

    print(A)
    
    print(Q)
    print(H)
    print(Q @ H @ Q.T)

if __name__ == '__main__':
    solve()
