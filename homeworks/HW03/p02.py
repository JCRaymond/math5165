#!/usr/bin/env python3.10

import numpy as np
import numpy.linalg as la
import time


class timer:
    def __init__(self):
        self.time = time.perf_counter()

    def get_time(self):
        return time.perf_counter() - self.time


rng = np.random.default_rng()


def grad_desc(grad, x_0, *, alpha=0.01, epsilon=1e-10, maxiters=1000000):
    x = x_0
    grad_x = grad(x)
    for i in range(maxiters):
        x = x - alpha * grad_x
        grad_x = grad(x)
        if la.norm(grad_x) < epsilon:
            return x, i + 1
    return x, maxiters


phi = (np.sqrt(5) + 1) / 2


def golden_search(f, s, e, epsilon=1e-10):
    while (e - s) > epsilon:
        delta = (e - s) / phi
        a = e - delta
        b = s + delta
        if f(a) < f(b):
            e = b
        else:
            s = a
    return (s + e) / 2


def grad_desc_opt(f, x_0, *, epsilon=1e-10, maxiters=1000000):
    x = x_0
    grad_x = f.grad(x)
    for i in range(maxiters):
        alpha_objective = lambda alpha: f(x - alpha * grad_x)
        alpha = golden_search(alpha_objective, 0, 1, epsilon=epsilon)
        x = x - alpha * grad_x
        grad_x = f.grad(x)
        if la.norm(grad_x) < epsilon:
            return x, i + 1
    return x, maxiters


class Rosenbrock:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, X):
        x, y = X
        c = self.a - x
        d = y - x * x
        return c * c + self.b * d * d

    def grad(self, X):
        x, y = X
        c = self.b * (y - x * x)
        return np.array((2 * (x - self.a) - 4 * x * c, 2 * c))


class Himmelblau:
    def __call__(self, X):
        x, y = X
        a = x * x + y - 11
        b = x + y * y - 7
        return a * a + b * b

    def grad(self, X):
        x, y = X
        a = 2 * (x * x + y - 11)
        b = 2 * (x + y * y - 7)
        return np.array((2 * a * x + b, a + 2 * y * b))


def solve():
    x_0 = np.zeros(2)
    a = 2
    epsilon = 1e-7
    for b in (0.1, 1, 10, 25, 50, 100):
        f = Rosenbrock(a, b)

        t = timer()
        x1, i1 = grad_desc(f.grad, x_0, alpha=0.001, epsilon=epsilon)
        t1 = t.get_time()

        t = timer()
        x2, i2 = grad_desc_opt(f, x_0, epsilon=epsilon)
        t2 = t.get_time()
        print(f'=== Rosenbrock({a},{b}) ===')
        print(f' - Basic -')
        print(f'   time: {t1:.4e}')
        print(f'  iters: {i1}')
        print(f'   f(x): {f(x1):.4e}')
        print(f'')
        print(f' - Optimal -')
        print(f'   time: {t2:.4e}')
        print(f'  iters: {i2}')
        print(f'   f(x): {f(x2):.4e}')
        print(f'')

    f = Himmelblau()

    t = timer()
    x1, i1 = grad_desc(f.grad, x_0, alpha=0.001, epsilon=epsilon)
    t1 = t.get_time()

    t = timer()
    x2, i2 = grad_desc_opt(f, x_0, epsilon=epsilon)
    t2 = t.get_time()

    print(f'=== Himmelblau ===')
    print(f' - Basic -')
    print(f'   time: {t1:.4e}')
    print(f'  iters: {i1}')
    print(f'      x: {x1}')
    print(f'   f(x): {f(x1):.4e}')
    print(f'')
    print(f' - Optimal -')
    print(f'   time: {t2:.4e}')
    print(f'  iters: {i2}')
    print(f'      x: {x2}')
    print(f'   f(x): {f(x2):.4e}')
    print(f'')


if __name__ == '__main__':
    solve()
