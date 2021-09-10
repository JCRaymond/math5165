#!/usr/bin/env python3.9

import numpy as np


# Returns matrices P, L, and U, where L is a square, lower triangular matrix,
# U in a matrix with the same shape as A, and is essentially upper triangular,
# P is a permutation matrix, and PA = LU.
def P_LU(A, epsilon=1e-10):
    L = np.eye(A.shape[0])
    U = np.matrix(A, dtype=np.float64)
    rows = list(range(A.shape[0]))
    piv_cols = []

    # Gaussian Elimination for U (REF, not RREF), keep track of modifications
    # to row for L, keep track of how rows are swapped for the eventual
    # computation of P
    for piv_col in range(U.shape[1]):
        piv_loc = len(piv_cols)

        # Find largest pivot position in this column
        piv_row = max(range(piv_loc, U.shape[0]),
                      key=lambda r: np.abs(U[r, piv_col]))
        piv_val = U[piv_row, piv_col]

        if np.abs(piv_val) < epsilon:
            continue
        piv_cols.append(piv_col)

        if piv_loc != piv_row:
            # Swap rows in U
            U[(piv_loc, piv_row), :] = U[(piv_row, piv_loc), :]

            # Fix L accordingly
            for col in range(piv_loc):
                L[(piv_loc, piv_row), col] = L[(piv_row, piv_loc), col]

            # Record which row ends up where in U
            rows[piv_loc], rows[piv_row] = rows[piv_row], rows[piv_loc]

        U[piv_loc, :piv_col] = np.zeros(piv_col)

        # Eliminate other rows
        for row in range(piv_loc + 1, U.shape[0]):
            multiplier = U[row, piv_col] / piv_val
            U[row, piv_col:] -= multiplier * U[piv_loc, piv_col:]
            L[row, piv_loc] = multiplier

    # Calculate P based on where the rows "end up" (based on rows)
    P = np.zeros((A.shape[0], A.shape[0]))
    for loc in enumerate(rows):
        P[loc] = 1

    return P, L, U


def solve():
    epsilon = 1e-10
    A = np.matrix([[2, 2, 3, 1], [4, 5, 6, 2], [1, 2, 4, 3], [8, 4, 4, 1]])

    P, L, U = P_LU(A)
    print('P:')
    print(P)
    print('L:')
    print(L)
    print('U:')
    print(U)
    print(f'PA = LU (within {epsilon} Frobenius norm):',
          np.linalg.norm(P @ A - L @ U, ord='fro') < epsilon)


if __name__ == '__main__':
    solve()
