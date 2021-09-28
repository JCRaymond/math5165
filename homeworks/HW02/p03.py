#!/usr/bin/env python3.9

import numpy as np
import numpy.linalg as la


def solve():
    A = np.matrix([[3, 0], [4, 5]])

    print()
    print('=== Original Matrix ===')
    print('A:')
    print(A)
    print('rank:', la.matrix_rank(A))

    U, s, Vt = la.svd(A)

    # Take first column of U, first singular value, and first row of Vt for
    # rank 1 approximation

    A_1 = s[0] * (U[:, 0] @ Vt[0, :])

    print()
    print('=== Rank 1 Approximation ===')
    print('A_1:')
    print(A_1)
    print('rank:', la.matrix_rank(A_1))
    print(f'∞-norm distance from A:', la.norm(A - A_1, ord=np.inf))

    A_2 = np.matrix([[2, 1.6], [5, 4]])

    print()
    print('=== Alternative Rank 1 Approximation ===')
    print('A_2:')
    print(A_2)
    print('rank:', la.matrix_rank(A_2))
    print(f'∞-norm distance from A:', np.linalg.norm(A - A_2, ord=np.inf))


if __name__ == '__main__':
    solve()
