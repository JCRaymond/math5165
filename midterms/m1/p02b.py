#!/usr/bin/env python3.9

import numpy as np
import numpy.linalg as la


def col(lst):
    return np.atleast_2d(np.array(lst)).T


def solve():
    A = np.matrix([[2, 0, 0], [0, 2, 1], [0, 1, 2]])

    A_inv = la.inv(A)
    print(A_inv)

    x_0 = col([0, 1, 3])
    x_1 = A_inv @ x_0
    x_2 = A_inv @ x_1

    print(x_0)
    print(x_1)
    print(x_2)

    x_n = x_2
    for i in range(100):
        x_n = A_inv @ x_n
    print(x_n)


if __name__ == '__main__':
    solve()
