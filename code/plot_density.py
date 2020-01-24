import json
from glob import glob
from random import sample

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

img_size = 128


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def fibonacci(max_value, start=50):
    num2 = start
    series = start
    seq = []
    num = 0
    while series < max_value:
        num += 1
        num1 = num2
        num2 = series
        series = num1 + num2
        seq.append(num2)

    return seq[:-1] + [max_value]


def split_list(l):
    max_value = len(l)
    start = 50
    seq = fibonacci(max_value, start=start)
    return [l[:i] for i in seq]


if __name__ == '__main__':

    print(fibonacci(max_value=2500))
    with open("../output/img_embeddings.json", 'r') as f:
        embeddings = json.load(f)

    batch_size = 16

    paths = list(sorted(glob('/media/ml/data_ml/dogs-vs-cats-redux-kernels-edition/train/*.jpg')))
    train, test = train_test_split(paths, random_state=1337, test_size=0.1)
    train, val = train_test_split(train, random_state=1337, test_size=0.1)

    train_embeddings = {x: embeddings[x.split("/")[-1]] for x in train}
    test_embeddings = {x: embeddings[x.split("/")[-1]] for x in test}
    val_embeddings = {x: embeddings[x.split("/")[-1]] for x in val}

    print(len(train_embeddings), len(test_embeddings), len(val_embeddings))

    # X_train = np.array([train_embeddings[x] for x in train])
    # X_val = np.array([val_embeddings[x] for x in val])
    # X_test = np.array([test_embeddings[x] for x in test])
    #
    # kdensity = KernelDensity(bandwidth=1)
    #
    # kdensity.fit(X_train)
    #
    # pred = kdensity.score_samples(X_val)
    #
    # plt.hist(pred, bins=50)
    # plt.savefig('hist.png')

    step_sizes = 50 * [50]

    indexes = list(range(len(train)))
    seed = sample(indexes, k=step_sizes[0])
    indexes = list(set(indexes) - set(seed))

    steps = [[train[i] for i in seed]]

    used_indexes = seed.copy()

    i = 0

    for step_size in step_sizes[1:]:
        print("len(indexes)", len(indexes))
        print(i)
        i += 1

        X_train = np.array([train_embeddings[train[i]] for i in used_indexes])
        X_left = np.array([train_embeddings[train[i]] for i in indexes])

        kdensity = KernelDensity(bandwidth=1)

        kdensity.fit(X_train)

        pred = kdensity.score_samples(X_left)

        plt.figure()
        plt.hist(pred, bins=50)
        plt.savefig('hist.png')

        step_value = sorted(pred.tolist())[step_size]

        print("step_value", step_value)

        step_indexes = [x for x, v in zip(indexes, pred.tolist()) if v < step_value]

        steps.append([train[i] for i in step_indexes])

        indexes = list(set(indexes) - set(step_indexes))
        used_indexes += step_indexes

    indexes_random = list(range(len(train)))
    seed_random = sample(indexes_random, k=step_sizes[0])
    indexes_random = list(set(indexes_random) - set(seed_random))

    steps_random = [[train[i] for i in seed_random]]

    used_indexes_random = seed_random.copy()

    i = 0

    for step_size in step_sizes[1:]:
        print("len(indexes_random)", len(indexes_random))
        print(i)
        i += 1

        step_indexes_random = sample(indexes_random, k=step_size)

        steps_random.append([train[i] for i in step_indexes_random])

        indexes_random = list(set(indexes_random) - set(step_indexes_random))
        used_indexes_random += step_indexes_random

    with open('../output/learning_steps.json', 'w') as f:
        data = {"active_learning_steps": split_list([train[i] for i in used_indexes]),
                "random_steps": split_list([train[i] for i in used_indexes_random]),
                "train": train,
                "val": val,
                "test": test}
        json.dump(data, f, indent=4)

    print(len(steps), len(steps_random))

    print([len(x) for x in split_list(used_indexes)])
    print([len(x) for x in split_list(used_indexes_random)])
