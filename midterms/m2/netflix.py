#/usr/bin/env python3.10

import os
import numpy as np
import pickle


class RatingsInfo:
    def __init__(self):
        self.num_movies = 0
        self.training_dir = ''
        self.user_id_map = {}


def get_ratingsinfo(training_dir, ratingsinfo_fname='./ratingsinfo.data'):
    if os.path.isfile(ratingsinfo_fname):
        print('Loading ratingsinfo from dump...')
        with open(ratingsinfo_fname, 'rb') as f:
            return pickle.load(f)

    ri = RatingsInfo()
    ri.num_movies = sum(1 for name in os.listdir(training_dir)
                        if os.path.isfile(os.path.join(training_dir, name)))
    ri.training_dir = training_dir

    for name in os.listdir(training_dir):
        fname = os.path.join(training_dir, name)
        with open(fname, 'r') as f:
            lines = iter(f)
            next(lines)  # Skip header
            for line in lines:
                user_id = int(line.partition(',')[0])
                if user_id not in ri.user_id_map:
                    ri.user_id_map[user_id] = len(ri.user_id_map)

    with open(ratingsinfo_fname, 'wb') as f:
        pickle.dump(ri, f)

    return ri


def get_ratings(ratings_info, ratings_fname='./ratings.data'):
    if os.path.isfile(ratings_fname):
        print('Loading ratings from dump...')
        with open(ratings_fname, 'rb') as f:
            return pickle.load(f)

    ri = ratings_info
    ratings = np.zeros((ri.num_movies, len(ri.user_id_map)), dtype=np.byte)

    for i, name in enumerate(os.listdir(ri.training_dir)):
        fname = os.path.join(ri.training_dir, name)
        m_idx = int(name.removeprefix('mv_').removesuffix('.txt')) - 1
        if i % 100 == 0:
            print(f'Progress... ({i}/{ri.num_movies})')
        with open(fname, 'r') as f:
            lines = iter(f)
            next(lines)  # Skip header
            for line in lines:
                user_id_s, rating_s, _ = line.strip().split(',')
                user_idx = ri.user_id_map[int(user_id_s)]
                rating = int(rating_s)
                ratings[m_idx, user_idx] = rating

    with open(ratings_fname, 'wb') as f:
        pickle.dump(ratings, f)

    return ratings


def get_train_test_split(ratings_info,
                         percentage,
                         test_train_loc='.',
                         train_name='train',
                         test_name='test'):
    train_fname = os.path.join(test_train_loc,
                               f'{train_name}_{percentage}.data')
    test_fname = os.path.join(test_train_loc, f'{test_name}_{percentage}.data')

    if os.path.isfile(train_fname) and os.path.isfile(test_fname):
        print('Loading train-test-split from dump...')
        with open(train_fname, 'rb') as f:
            train = pickle.load(f)
        with open(test_fname, 'rb') as f:
            test = pickle.load(f)
        return train, test

    ratings = get_ratings(ratings_info)
    rs, cs = np.nonzero(ratings)
    nz_entries = len(rs)
    test_idxs = np.arange(nz_entries)
    rng = np.random.default_rng()
    rng.shuffle(test_idxs)
    test_idxs = test_idxs[:int(percentage * nz_entries)]
    rs = rs[test_idxs]
    cs = cs[test_idxs]
    test_ratings = ratings[rs, cs]
    ratings[rs, cs] = 0

    with open(train_fname, 'wb') as f:
        train = pickle.dump(ratings, f)
    with open(test_fname, 'wb') as f:
        test = pickle.dump((rs, cs, test_ratings), f)

    return train, test


def blockmul(A, B, a_step=350, b_step=None):
    n, m = A.shape
    if m != B.shape[0]:
        return None
    m, k = B.shape

    if b_step is None:
        b_step = a_step

    C = np.empty((n, k))

    temp1 = np.empty((a_step, m), dtype=np.double)
    temp2 = np.empty((m, b_step), dtype=np.double)
    for i in range(0, n, a_step):
        ui = i + a_step
        ui = ui if ui < n else n
        di = ui - i
        temp1[:di, :] = A[i:ui, :]
        for j in range(0, k, b_step):
            #print(f'\tIter {j//b_step + 1}/{k//b_step+(0 if k%b_step==0 else 1)}')
            uj = j + b_step
            uj = uj if uj < k else k
            dj = uj - j
            temp2[:, :dj] = B[:, j:uj]
            #print(f'C[{i}:{ui}, {j}:{uj}] = temp1[:{di}, :] @ temp2[:, :{dj}]')
            C[i:ui, j:uj] = temp1[:di, :] @ temp2[:, :dj]
    del temp1
    del temp2

    return C


def get_B(A, B_fname='./B.data'):
    if os.path.isfile(B_fname):
        print('Loading B from dump...')
        with open(B_fname, 'rb') as f:
            return pickle.load(f)

    print('Calculating B...')
    B = blockmul(A, A.T, a_step=350, b_step=350)

    with open(B_fname, 'wb') as f:
        pickle.dump(B, f)

    return B


def svd(A, *, B=None, k=None, s_min=None, epsilon=1e-12):
    n, m = A.shape
    if n > m:
        U, S, Vt = svd(A.T, k=k, s_min=s_min)
        return Vt.T, S, U.T
    if k is None:
        k = n
    if s_min is None or s_min < 0:
        s_min = 0

    S = np.empty(k)
    U = np.empty((n, k))
    V = np.empty((m, k))

    if B is None:
        B = blockmul(A, A.T, a_step=350, b_step=350)

    for i in range(k):
        if os.path.isfile('./STOPSVD'):
            break
        print(f'Finding singular value {i+1}/{k}...')
        # Take initial vector as first nonzero column of B
        for j in range(n):
            if np.linalg.norm(B[:, j]) != 0:
                u = np.atleast_2d(np.array(B[:, j])).T
                break
        else:
            break

        u = u / np.linalg.norm(u)
        dist = 1 + epsilon
        print(f'\tConverging to eigenvector...')
        while dist > epsilon:
            u_old = u
            u = B @ u
            u = u / np.linalg.norm(u)

            dist = np.linalg.norm(u - u_old)
            print(f'\t\tdist: {dist}')

        print('\tCalculating v...')
        v_unnormed = blockmul(A.T, u, a_step=16384, b_step=1)
        s = np.linalg.norm(v_unnormed)
        print(f'\tFound singular value: {s}')
        if s <= s_min:
            break
        v = v_unnormed / s

        U[:, i:i + 1] = u
        V[:, i:i + 1] = v
        S[i] = s

        print(f'\tUpdate B...')
        u *= s
        B -= u @ u.T
    else:
        i = k
    U = U[:, :i]
    V = V[:, :i]
    S = S[:i]
    return U, S, V.T


def main():
    print('Getting info about ratings...')
    ri = get_ratingsinfo('./data/training_set')
    print('Getting training and test data...')
    train, test = get_train_test_split(ri, 0.01)

    print('Begin svd:')
    U, S, Vt = svd(train, B=get_B(train), k=1000, s_min=0.05)

    with open('U.data', 'wb') as f:
        pickle.dump(U, f)
    with open('S.data', 'wb') as f:
        pickle.dump(S, f)
    with open('Vt.data', 'wb') as f:
        pickle.dump(Vt, f)


def test():
    A = np.matrix([[1, 3, 5, 6, 1, 9], [2, 8, 4, 3, 0, 8], [3, 1, 7, -3, 8, 6],
                   [10, 1, 9, 4, 8, 4], [16, 1, 0, 0, 3, -1]])

    U, S, Vt = svd(A)
    U_, S_, Vt_ = np.linalg.svd(A)
    print(A)
    print((U * S) @ Vt)


if __name__ == '__main__':
    main()
    #test()
