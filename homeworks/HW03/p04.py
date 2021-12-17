#!/usr/bin/env python3.10

import numpy as np
import numpy.linalg as la
import time

rng = np.random.default_rng()


class timer:
    def __init__(self):
        self.time = time.perf_counter()

    def get_time(self):
        return time.perf_counter() - self.time


def rand_mult(A, B, c, *, p=None):
    m, n = A.shape
    if B.shape[0] != n:
        return None
    n, l = B.shape
    if p is None or p == 'opt':
        p = np.sum((A * A), axis=0) * np.sum((B * B), axis=1)
        p = np.sqrt(p)
        p /= la.norm(p, ord=1)
    elif p == 'unif':
        p = np.ones(n, dtype=np.float64)
        p /= la.norm(p, ord=1)

    idxs = np.arange(n)
    random_idxs = rng.choice(idxs, size=c, p=p)
    C = np.empty((m, l))
    for i in random_idxs:
        C += np.outer(A[:, i], B[i, :]) / (c * p[i])
    return C


def solve():
    trials = 10
    print('k, full, unif, unif_err, opt, opt_err')
    for k in range(4, 11):
        m = l = n = 1 << k
        c = int(np.sqrt(n)) + 1

        t0 = 0
        t1 = 0
        e1 = 0
        t2 = 0
        e2 = 0
        for t in range(trials):
            A = rng.random((m, n))
            B = rng.random((n, l))

            t = timer()
            C = A @ B
            t0 += t.get_time()

            t = timer()
            C_unif = rand_mult(A, B, c, p='unif')
            t1 += t.get_time()
            e1 += la.norm(C - C_unif, ord='fro')

            t = timer()
            C_opt = rand_mult(A, B, c)
            t2 += t.get_time()
            e2 += la.norm(C - C_opt, ord='fro')

        t0 /= trials
        t1 /= trials
        e1 /= trials
        t2 /= trials
        e2 /= trials

        print(f'{k}, {t0:.4e}, {t1:.4e}, {e1:.4e}, {t2:.4e}, {e2:.4e}')


if __name__ == '__main__':
    solve()
