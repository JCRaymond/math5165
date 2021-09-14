#!/usr/bin/env python3.9

import numpy as np


# Returns a matrix R which is A in reduced row-echelon form, along with
# a list of indices representing which columns of A have pivots.
def ref(A, epsilon=1e-10):
    X = np.matrix(A, dtype=np.float64)
    num_pivs = 0
    for piv_col in range(X.shape[1]):
        print(X)
        piv_loc = num_pivs
        # Find largest pivot position in this column
        piv_row = max(range(piv_loc, X.shape[0]),
                      key=lambda r: np.abs(X[r, piv_col]))
        piv_val = X[piv_row, piv_col]

        if np.abs(piv_val) < epsilon:
            continue
        num_pivs += 1

        if piv_loc != piv_row:
            X[(piv_loc, piv_row), :] = X[(piv_row, piv_loc), :]

        X[piv_loc, :piv_col] = np.zeros(piv_col)
        normed_piv_row = X[piv_loc, piv_col:] / piv_val

        # Eliminate other rows
        for row in range(X.shape[0]):
            if row != piv_loc:
                X[row, piv_col:] -= X[row, piv_col] * normed_piv_row

    return X


def QD(A, epsilon=1e-10):
   if A.shape != A.T.shape or np.linalg.norm(A - A.T, ord=np.inf) > epsilon:
      return None

   D = ref(A)
   return [R[i,i] for i in range(R.shape[0])]
   ||(A-lI)x|| = 0 = <x, h>

   A-lIx = 0

   (A-lI)x = 0



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
    print(sym_evals(A))
