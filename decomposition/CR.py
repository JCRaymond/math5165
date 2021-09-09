#!/usr/bin/env python3.9

import numpy as np


# Returns a matrix R which is A in reduced semi-row-echelon form, along with
# a list of tuples representing the indices of the pivots in A. It is
# similar to reduced row-echelon form, however no rows are swapped.
def rsref(A, epsilon=1e-10):
    X = np.matrix(A, dtype=np.float64)
    piv_rows = set()
    piv_poss = []
    for piv_col in range(X.shape[1]):
        # Find largest pivot position in this column
        piv_row = max(set(range(X.shape[0])) - piv_rows,
                      key=lambda r: np.abs(X[r, piv_col]))
        piv_val = X[piv_row, piv_col]

        if np.abs(piv_val) < epsilon:
            continue
        piv_rows.add(piv_row)
        piv_poss.append((piv_row, piv_col))

        X[piv_row, :piv_col] = np.zeros(piv_col)
        X[piv_row, piv_col:] /= piv_val

        # Eliminate other rows
        for row in range(X.shape[0]):
            if row != piv_row:
                X[row, piv_col:] -= X[row, piv_col] * X[piv_row, piv_col:]

    return X, piv_poss


# Returns a matrix R which is A in reduced row-echelon form, along with
# a list of indices representing which columns of A have pivots.
def rref(A, epsilon=1e-10):
    X = np.matrix(A, dtype=np.float64)
    piv_cols = []
    for piv_col in range(X.shape[1]):
        print(X)
        piv_loc = len(piv_cols)
        # Find largest pivot position in this column
        piv_row = max(range(piv_loc, X.shape[0]),
                      key=lambda r: np.abs(X[r, piv_col]))
        piv_val = X[piv_row, piv_col]

        if np.abs(piv_val) < epsilon:
            continue
        piv_cols.append(piv_col)

        if piv_loc != piv_row:
            X[(piv_loc, piv_row), :] = X[(piv_row, piv_loc), :]

        X[piv_loc, :piv_col] = np.zeros(piv_col)
        X[piv_loc, piv_col:] /= piv_val

        # Eliminate other rows
        for row in range(X.shape[0]):
            if row != piv_loc:
                X[row, piv_col:] -= X[row, piv_col] * X[piv_loc, piv_col:]

    return X, piv_cols


# Return matrices C and R such that C consists of independent columns of A,
# and A = CR
def CR(A):
    X, piv_cols = rref(A)
    C = A[:, piv_cols]
    R = X[:len(piv_cols), :]
    return C, R


if __name__ == '__main__':

    # A = np.matrix([[0, 1, 1], [2, 2, 2], [3, 3, 3]])
    #A = np.random.rand(6, 6)
    #A[:, 4] = 3 * A[:, 2] + 2 * A[:, 1] - A[:, 0]
    #A[:, 3] = A[:, 1] - A[:, 0]
    A = np.matrix([[-1, 1, 0, 0, 0], [1, 0, -1, 0, 0], [0, 1, -1, 0, 0],
                   [-1, 0, 0, 1, 0], [0, -1, 0, 0, 1], [0, 0, 1, -1, 0],
                   [0, 0, -1, 0, 1], [0, 0, 0, 1, -1]])

    print(A)
    C, R = CR(A)
    print(C)
    print(R)
    print(C * R)
    print(np.linalg.norm(A - C * R, ord='fro') < 1e-6)
