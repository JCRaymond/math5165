#!/usr/bin/env python3.9

import numpy as np


# Returns a matrix R which is A in reduced row-echelon form, along with
# a list of indices representing which columns of A have pivots.
def rref(A, epsilon=1e-10):
    X = np.matrix(A, dtype=np.float64)
    piv_cols = []
    for piv_col in range(X.shape[1]):
        piv_loc = len(piv_cols)
        # Find largest pivot position in this column
        piv_row = max(range(piv_loc, X.shape[0]),
                      key=lambda r: np.abs(X[r, piv_col]))

        if np.abs(X[piv_row, piv_col]) < epsilon:
            continue
        piv_cols.append(piv_col)

        if piv_loc != piv_row:
            X[(piv_loc, piv_row)] = X[(piv_row, piv_loc)]

        X[piv_loc, :piv_col] = np.zeros(piv_col)
        X[piv_loc, piv_col:] /= X[piv_loc, piv_col]

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


# Returns a matrix R which is A in reduced row-echelon form, along with
# a list of indices representing which columns of A have pivots.
def CMR(A, epsilon=1e-10):
    X = np.matrix(A, dtype=np.float64)
    rows = list(range(A.shape[0]))
    M = np.eye(A.shape[0])
    piv_cols = []

    # Gaussian Elimination along with the calculation of the inverse matrix M
    # If A is invertible, then C = A, M = A^{-1}, R = A. Otherwise, a portion
    # of the computed matrix M is used as the mixing matrix.
    for piv_col in range(X.shape[1]):
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
            M[(piv_loc, piv_row), :] = M[(piv_row, piv_loc), :]
            rows[piv_loc], rows[piv_row] = rows[piv_row], rows[piv_loc]

        X[piv_loc, :piv_col] = np.zeros(piv_col)
        X[piv_loc, piv_col:] /= piv_val
        M[piv_loc, :] /= piv_val

        # Eliminate other rows
        for row in range(X.shape[0]):
            if row != piv_loc:
                factor = X[row, piv_col]
                X[row, piv_col:] -= factor * X[piv_loc, piv_col:]
                M[row, :] -= factor * M[piv_loc, :]

    good_rows = sorted(rows[:len(piv_cols)])

    C = A[:, piv_cols]
    M = M[:len(piv_cols), :][:, good_rows]
    R = A[good_rows, :]

    return C, M, R


def solve():
    A = np.matrix([[1, 3, 8], [1, 2, 6], [0, 1, 2]])
    print('A:')
    print(A)

    print()
    print()

    C_, R_ = CR(A)
    print('CR Decomposition:')
    print('C:')
    print(C_)
    print('R:')
    print(R_)
    print('CR:')
    print(C_ @ R_)

    print()
    print()

    C, M, R = CMR(A)
    print('CMR Decomposition:')
    print('C:')
    print(C)
    print('M:')
    print(M)
    print('R:')
    print(R)
    print('CMR:')
    print(C @ M @ R)


if __name__ == '__main__':
    solve()
