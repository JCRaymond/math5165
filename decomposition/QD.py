#!/usr/bin/env python3.9

import numpy as np

def givens_rot(n, i, a, j, b, epsilon=1e-10):
    G = np.eye(n)
    r = np.hypot(a, b)
    if np.abs(b) < epsilon or np.abs(r) < epsilon:
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
def QR(A, epsilon=1e-10):
    n = A.shape[0]
    R = np.matrix(A)
    Q = np.eye(n)
    for col in range(R.shape[1] - 1):
        for row in range(R.shape[0] - 1, col, -1):
            G_i = givens_rot(n, row - 1, R[row - 1, col], row, R[row, col])
            Q = Q @ G_i.T
            R = G_i @ R
    return Q, R

# Returns a non-zero vector in the null space of A if one exists, returns
# the zero vector otherwise.
def null_vec(A, epsilon=1e-10):
    X = np.matrix(A, dtype=np.float64)
    piv_cols = {}
    for piv_col in range(X.shape[1]):
        piv_loc = len(piv_cols)
        # Find largest pivot position in this column
        piv_row = max(range(piv_loc, X.shape[0]),
                      key=lambda r: np.abs(X[r, piv_col]))
        piv_val = X[piv_row, piv_col]

        if np.abs(piv_val) < epsilon:
            continue
        piv_cols[piv_col] = piv_row

        if piv_loc != piv_row:
            X[(piv_loc, piv_row), :] = X[(piv_row, piv_loc), :]

        X[piv_loc, :piv_col] = np.zeros(piv_col)
        X[piv_loc, piv_col:] /= piv_val

        # Eliminate other rows
        for row in range(X.shape[0]):
            if row != piv_loc:
                X[row, piv_col:] -= X[row, piv_col] * X[piv_loc, piv_col:]
    
    vec = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        if i not in piv_cols:
            vec[i] = 1
            continue

        row = piv_cols[i]
        vec[i] -= np.sum(X[piv_row, i+1:X.shape[1]])
    vec = np.atleast_2d(vec).T
    vec_norm = np.linalg.norm(vec)
    if np.abs(vec_norm) < epsilon:
        return vec
    return vec / vec_norm

def QD(A, epsilon=1e-10):
   if A.shape != A.T.shape or np.linalg.norm(A - A.T, ord=np.inf) > epsilon:
      return None

   Q = np.eye(A.shape[0])
   D = np.matrix(A)
   dist = 1 + epsilon
   while dist > epsilon:
       Q_, R = QR(D)
       D_new = R @ Q_
       Q = Q @ Q_

       dist = np.linalg.norm(D - D_new, ord='fro')
       D = D_new
   
   return Q, D

if __name__ == '__main__':

    A = np.matrix([[1, 2, 3], [4, 5, 4], [3, 2, 1]])
    #A = np.random.rand(6, 6)
    #A[:, 4] = 3 * A[:, 2] + 2 * A[:, 1] - A[:, 0]
    #A[:, 3] = A[:, 1] - A[:, 0]
    #A = np.matrix([[-1, 1, 0, 0, 0], [1, 0, -1, 0, 0], [0, 1, -1, 0, 0],
    #               [-1, 0, 0, 1, 0], [0, -1, 0, 0, 1], [0, 0, 1, -1, 0],
    #               [0, 0, -1, 0, 1], [0, 0, 0, 1, -1]])
    A = (A + A.T)/2

    print(A)
    Q, D = QD(A)
    print(Q)
    print(D)
    print(Q @ D @ Q.T)
    print(Q @ Q.T)
