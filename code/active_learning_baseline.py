import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D


def entropy(l):
    return -sum([x * np.log(np.clip(x, 1e-12, 1 - 1e-12)) for x in l])


def get_model():
    nclass = 5
    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    #model.summary()
    return model


if __name__ == "__main__":

    df_train = pd.read_csv("../input/mitbih_train.csv", header=None)
    df_test = pd.read_csv("../input/mitbih_test.csv", header=None)

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    print(X.shape)

    step_sizes = [128]*20

    unused_samples = list(range(X.shape[0]))
    step = np.random.choice(unused_samples, size=512).tolist()
    used_samples = step
    unused_samples = list(set(unused_samples) - set(step))

    results = []

    rnd = np.random.randint(1, 100000)

    for i, step_size in enumerate(step_sizes):
        X_used = X[used_samples, ...]
        Y_used = Y[used_samples, ...]

        X_unused = X[unused_samples, ...]
        Y_unused = Y[unused_samples, ...]

        model = get_model()

        model.fit(X_used, Y_used, epochs=45, verbose=1, batch_size=32)

        pred_ununsed = model.predict(X_unused).tolist()
        entr = [entropy(l) for l in pred_ununsed]
        threshold = sorted(entr, reverse=True)[step_size]

        pred_test = model.predict(X_test)
        pred_test = np.argmax(pred_test, axis=-1)

        f1 = f1_score(Y_test, pred_test, average="macro")

        acc = accuracy_score(Y_test, pred_test)

        results.append({"size": X_used.shape[0], "accuracy": acc, "f1": f1})

        print(results[-1])

        step = [x for x, v in zip(unused_samples, entr) if v >= threshold]
        used_samples += step
        unused_samples = list(set(unused_samples) - set(step))

        with open('../output/active_learning_performance_%s.json'%float(rnd), 'w') as f:
            json.dump(results, f, indent=4)
