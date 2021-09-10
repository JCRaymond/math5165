#!/usr/bin/env python3.9

import numpy as np


def col(vec):
    return np.atleast_2d(vec).T


def solve():
    A = np.matrix([[-1, 1, 0, 0, 0], [1, 0, -1, 0, 0], [0, 1, -1, 0, 0],
                   [-1, 0, 0, 1, 0], [0, -1, 0, 0, 1], [0, 0, 1, -1, 0],
                   [0, 0, -1, 0, 1], [0, 0, 0, 1, -1]])

    v_1 = col(np.ones(5))
    print('v_1 in N(A): ', np.linalg.norm(A * v_1, ord=np.inf) < 1e-10)

    L = np.matrix([[1, 1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1],
                   [0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, -1, 0]])
    print('L row 1 in N(A.T): ', np.linalg.norm(L[0, :] * A) < 1e-10)
    print('L row 2 in N(A.T): ', np.linalg.norm(L[1, :] * A) < 1e-10)
    print('L row 3 in N(A.T): ', np.linalg.norm(L[2, :] * A) < 1e-10)
    print('L row 4 in N(A.T): ', np.linalg.norm(L[3, :] * A) < 1e-10)


if __name__ == '__main__':
    solve()
