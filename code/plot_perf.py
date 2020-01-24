import json
from glob import glob
from random import sample, shuffle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

if __name__ == '__main__':
    with open("../output/active_learning_performance.json", 'r') as f:
        data = json.load(f)

    accuracies = [x['accuracy'] for x in data]
    size = [x['size'] for x in data]

    with open("../output/random_performance.json", 'r') as f:
        data_random = json.load(f)

    accuracies_random = [x['accuracy']for x in data_random]
    size_random = [x['size'] for x in data_random]

    plt.figure()
    plt.plot(size, accuracies, label="Active Learning")
    plt.plot(size_random, accuracies_random, label="Random")
    plt.legend()
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")

    plt.show()
