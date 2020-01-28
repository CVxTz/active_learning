import json
from glob import glob
from random import sample, shuffle

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    folder = "ecg"
    len_ = 20

    accuracies = 0
    size = 0
    cnt = 0
    for p in glob(f"../output/{folder}/active_learning_performance*.json"):
        with open(p, 'r') as f:
            data = json.load(f)

        if len(data) == len_:
            accuracies += np.array([x['f1'] for x in data])
            size += np.array([x['size'] for x in data])
            cnt += 1

    accuracies = accuracies/cnt
    size = size/cnt

    accuracies_random = 0
    size_random = 0
    cnt_random = 0
    for p in glob(f"../output/{folder}/random_performance*.json"):
        with open(p, 'r') as f:
            data = json.load(f)

        if len(data) == len_:
            accuracies_random += np.array([x['f1'] for x in data])
            size_random += np.array([x['size'] for x in data])
            cnt_random += 1

    accuracies_random = accuracies_random/cnt_random
    size_random = size_random/cnt_random

    plt.figure()
    plt.plot(size, accuracies, label="Active Learning")
    plt.plot(size_random, accuracies_random, label="Random")
    plt.legend()
    plt.xlabel("Training Size")
    plt.ylabel("F1 score (macro)")
    plt.title(folder.upper())

    plt.savefig(folder+".png")
