#!/usr/bin/env python3.10

import numpy as np
import numpy.linalg as la
import time
from sys import stderr


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
        self.n = X.shape[1]

    def grad_i(self, i, args):
        alpha = args[0]
        beta = args[1:]
        grad = np.empty(self.d + 1)
        grad[0] = 1
        grad[1:] = self.X[:, i]
        z = -alpha - np.dot(beta, self.X[:, i])
        p = 1 / (1 + np.exp(z))
        grad *= p - self.y[i]
        return grad

    # Do not use!
    def grad(self, args):
        grad = self.grad_i(0, args)
        for i in range(1, self.n):
            grad += self.grad_i(i, args)
        grad /= self.n
        return grad


class LogisticPredictor:
    def __init__(self, args):
        self.alpha = args[0]
        self.beta = args[1:]

    def __call__(self, x):
        z = -self.alpha - np.dot(self.beta, x)
        p = 1 / (1 + np.exp(z))
        return 1 if p >= 0.5 else 0


def stoch_grad_desc(cost,
                    x_0,
                    *,
                    train_perc=0.01,
                    alpha=0.005,
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
    grad_x /= step_samples
    for i in range(maxiters):
        if i % 1000 == 0:
            print(i, file=stderr)
        x = x - alpha * grad_x
        step_idxs = rng.choice(idxs,
                               step_samples,
                               replace=False,
                               shuffle=False)
        grad_x = cost.grad_i(step_idxs[0], x)
        for j in range(1, step_samples):
            grad_x += cost.grad_i(step_idxs[j], x)
        grad_x /= step_samples
        if la.norm(grad_x) < epsilon:
            if stable_iters >= needed_stable_iters:
                return x, i + 1
            stable_iters += 1
        else:
            stable_iters = 0
    return x, maxiters


def stoch_grad_desc_ridge(cost,
                          x_0,
                          *,
                          mu=1,
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
    grad_x /= step_samples
    grad_x -= mu * x  # Ridge term
    for i in range(maxiters):
        if i % 1000 == 0:
            print(i, file=stderr)
        x = x - alpha * grad_x
        step_idxs = rng.choice(idxs,
                               step_samples,
                               replace=False,
                               shuffle=False)
        grad_x = cost.grad_i(step_idxs[0], x)
        for i in range(1, step_samples):
            grad_x += cost.grad_i(step_idxs[i], x)
        grad_x /= step_samples
        grad_x -= mu * x  # Ridge term
        if la.norm(grad_x) < epsilon:
            if stable_iters >= needed_stable_iters:
                return x, i + 1
            stable_iters += 1
        else:
            stable_iters = 0
    return x, maxiters


def get_data(fname):
    X = []
    y = []
    with open(fname, 'r') as f:
        for line in f:
            age, _, fnlwgt, _, education_num, _, _, _, _, sex, capital_gain, capital_loss, hours_per_week, _, salary = map(
                str.strip, line.split(','))
            age = int(age)
            fnlwgt = int(fnlwgt)
            education_num = int(education_num)
            sex = 0 if sex == 'Male' else 1
            capital_gain = int(capital_gain)
            capital_loss = int(capital_loss)
            hours_per_week = int(hours_per_week)
            salary = 0 if salary == '<=50K' else 1
            X.append(
                np.array((age, fnlwgt, education_num, sex, capital_gain,
                          capital_loss, hours_per_week)))
            y.append(salary)

    X = np.array(X, dtype=np.float64)
    X = (X - X.min(axis=0))
    X /= X.max(axis=0)
    y = np.array(y)
    return X.T, y


def test():
    X = np.array(((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))).T
    y = np.array((0, 1, 1, 0, 0, 1))
    cost = LogisticRegressionCost(X, y)
    initial_weights = np.zeros(cost.d + 1)
    optimal_weights, iters = stoch_grad_desc(cost, initial_weights, alpha=0.1)
    print(optimal_weights)
    pred = LogisticPredictor(optimal_weights)
    for i in range(X.shape[1]):
        print(pred(X[:, i]), y[i])


def solve():
    X, y = get_data('adult_train.csv')
    cost = LogisticRegressionCost(X, y)
    initial_weights = np.ones(cost.d + 1)
    optimal_weights, iters = stoch_grad_desc(cost,
                                             initial_weights,
                                             train_perc=0.02,
                                             epsilon=1e-5,
                                             alpha=0.02,
                                             maxiters=100000)
    print(optimal_weights)
    print('Training Iterations:', iters)
    pred = LogisticPredictor(optimal_weights)
    train_acc = 0
    for i in range(X.shape[1]):
        if pred(X[:, i]) == y[i]:
            train_acc += 1
    train_acc /= X.shape[1]
    print('Training Accuracy:', train_acc)

    X_test, y_test = get_data('adult_test.csv')
    test_acc = 0
    for i in range(X_test.shape[1]):
        if pred(X_test[:, i]) == y_test[i]:
            test_acc += 1
    test_acc /= X_test.shape[1]
    print('Test Accuracy:', test_acc)


if __name__ == '__main__':
    #test()
    solve()
