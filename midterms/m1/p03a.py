#!/usr/bin/env python3.9

import numpy as np
import numpy.linalg as la


def col(lst):
    return np.atleast_2d(np.array(lst)).T


# Returns a householder matrix H such that Hu = ||u|| e_i
def Householder(u, i):
    n = u.shape[0]
    u_norm = la.norm(u)
    h = np.matrix(u)
    h[i] += u_norm
    return np.eye(n) - (2 / (h.T @ h).item()) * (h @ h.T)


def solve():
    x = col([2., 1, 2])
    # 0 indexed, so i=1 will make second entry non-zero
    H = Householder(x, 1)
    v = H @ x
    print('Householder matrix H:')
    print(H)
    print()

    print('v:')
    print(v)


if __name__ == '__main__':
    solve()
