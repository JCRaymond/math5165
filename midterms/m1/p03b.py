#!/usr/bin/env python3.9

import numpy as np
import numpy.linalg as la


def col(lst):
    return np.atleast_2d(np.array(lst)).T


# Returns a Givens matrix G which will rotate an n-dimensional vector
# with ith component a and jth component b so that the ith component is
# sqrt(a^2 + b^2) and jth component is 0.
def Givens(n, i, a, j, b, epsilon=1e-10):
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


def solve():
    x = col([2., 1, 2])
    # 0 indexed, so this will rotate 1st component into second component
    J_2 = Givens(3, 1, x[1], 0, x[0])
    x_prime = J_2 @ x
    J_1 = Givens(3, 1, x_prime[1], 2, x_prime[2])
    u = J_1 @ J_2 @ x

    print('Givens matrix J_1:')
    print(J_1)
    print()

    print('Givens matrix J_2:')
    print(J_2)
    print()

    print('u:')
    print(u)


if __name__ == '__main__':
    solve()
