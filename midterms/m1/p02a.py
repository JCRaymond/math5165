#!/usr/bin/env python3.9

import numpy as np
import numpy.linalg as la


def col(lst):
    return np.atleast_2d(np.array(lst)).T


def solve():
    A = np.matrix([[2, 0, 0], [0, 2, 1], [0, 1, 2]])
    print('A:')
    print(A)
    print()

    x_0 = col([0, 1, 3])
    x_1 = A @ x_0
    x_2 = A @ x_1

    print('xs:')
    print(x_0)
    print(x_1)
    print(x_2)
    print()

    print('bs:')
    print(x_0 / la.norm(x_0))
    print(x_1 / la.norm(x_1))
    print(x_2 / la.norm(x_2))
    print()


if __name__ == '__main__':
    solve()
