#!/usr/bin/env python3.10

import numpy as np
import numpy.linalg as la
from PIL import Image

rng = np.random.default_rng()


def k_means(X, K):
    n, m = X.shape
    U = X[:,
          rng.choice(np.arange(m), replace=False, size=K)].astype(np.float64)
    deltas = np.empty((m, n))
    R_vals = np.empty((K, m))
    R = np.zeros(m, dtype=np.int32)
    new_R = np.empty(m, dtype=np.int32)

    R_changed = True
    while R_changed:

        # Update R
        for k in range(K):
            np.subtract(X.T, U[:, k], out=deltas)
            deltas *= deltas
            np.sum(deltas, out=R_vals[k, :], axis=1)
        np.argmin(R_vals, out=new_R, axis=0)
        R_changed = np.any(R != new_R)
        R, new_R = new_R, R

        if not R_changed:
            break

        # Update U
        for k in range(K):
            idxs = np.argwhere(R == k)
            l = len(idxs)
            idxs = idxs.reshape((l, ))
            np.sum(X[:, idxs], out=U[:, k], axis=1)
            if l > 0:
                U[:, k] /= l

    return U, R


def test():
    X = np.array([[3, 1], [3, 0], [0, 1], [0, 0]]).T
    U, R = k_means(X, 2)
    print(U.T)
    print(R)


def simplify(pic, colors):
    rgb = np.asarray(pic)
    rgb_dtype = rgb.dtype
    N, M, d = rgb.shape
    rgb = rgb.reshape(N * M, d).T
    U, R = k_means(rgb, colors)
    U = U.astype(rgb_dtype)
    rgb = U[:, R]
    rgb = rgb.T.reshape(N, M, d)
    return Image.fromarray(rgb)


def solve():
    print('Loading Image...')
    pic = Image.open('p01-2.jpg')
    for i in (2, 4, 8, 16, 32):
        print(f'Simplifying to {i} colors...')
        pic2 = simplify(pic, i)
        pic2.save(f'p01-2-k{i}.jpg')


if __name__ == '__main__':
    #test()
    solve()
