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


if __name__ == '__main__':
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

    step_size = 1000

    indexes = list(range(len(train)))
    seed = sample(indexes, k=step_size)
    indexes = list(set(indexes) - set(seed))

    steps = [[train[i] for i in seed]]

    used_indexes = seed.copy()

    i = 0

    while len(indexes) > step_size:
        print("len(indexes)", len(indexes))
        print(i)
        i += 1

        X_train = np.array([train_embeddings[train[i]] for i in used_indexes])
        X_left = np.array([train_embeddings[train[i]] for i in indexes])

        kdensity = KernelDensity(bandwidth=1)

        kdensity.fit(X_train)

        pred = kdensity.score_samples(X_left)

        plt.hist(pred, bins=50)
        plt.savefig('hist.png')

        step_value = sorted(pred.tolist())[step_size]

        print("step_value", step_value)

        step_indexes = [x for x, v in zip(indexes, pred.tolist()) if v < step_value]

        steps.append([train[i] for i in step_indexes])

        indexes = list(set(indexes) - set(step_indexes))
        used_indexes += step_indexes

    steps.append([train[i] for i in indexes])

    with open('../output/learning_steps.json', 'w') as f:
        data = {"active_learning_steps": steps,
                "random_steps": list(chunker(train, size=step_size)),
                "train": train,
                "val": val,
                "test": test}
        json.dump(data, f, indent=4)
