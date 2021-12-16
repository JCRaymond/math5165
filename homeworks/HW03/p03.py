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


class LogisticRegressionCost:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.d = X.shape[0]
        sefl.n = X.shape[1]

    def grad_i(self, i, args):
        alpha = args[0]
        beta = args[1:]
        grad = np.empty(self.d + 1)
        grad[0] = 1
        grad[1:] = self.X[:, i]
        z = -alpha - np.dot(beta, self.X[:, i])
        p = 1 / (1 + np.exp(z))
        grad *= self.y[i] - p
        return grad

    # Do not use!
    def grad(self, args):
        grad = self.grad_i(0, args)
        for i in range(1, self.n):
            grad += self.grad_i(i, args)
        return grad


def stoch_grad_desc(cost,
                    x_0,
                    *,
                    train_perc=0.01,
                    alpha=0.001,
                    epsilon=1e-7,
                    maxiters=1000000):
    idxs = np.arange(cost.n)
    step_samples = int(train_perc * cost.n)
    if step_samples < 1:
        step_samples = 1

    needed_stable_iters = int(np.sqrt(1 / train_perc)) + 1
    stable_iters = 0

    x = x_0
    step_idxs = rng.choice(idxs, step_samples, replace=False, shuffle=False)
    grad_x = cost.grad_i(step_idxs[0], x)
    for i in range(1, step_samples):
        grad_x += cost.grad_i(step_idxs[i], x)
    for i in range(maxiters):
        x = x - alpha * grad_x
        step_idxs = rng.choice(idxs,
                               step_samples,
                               replace=False,
                               shuffle=False)
        grad_x = cost.grad_i(step_idxs[0], x)
        for i in range(1, step_samples):
            grad_x += cost.grad_i(step_idxs[i], x)
        if la.norm(grad_x) < epsilon:
            if stable_iters >= needed_stable_iters:
                return x, i + 1
            stable_iters += 1
        else:
            stable_iters = 0
    return x, maxiters


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
