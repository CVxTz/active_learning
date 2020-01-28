import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution2D, MaxPool2D, GlobalMaxPool2D
from keras.datasets import mnist
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.backend import clear_session


def entropy(l):
    return -sum([x * np.log(np.clip(x, 1e-12, 1 - 1e-12)) for x in l])


def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))


def get_model():
    nclass = 10
    inp = Input(shape=(None, None, 1))
    img_1 = Convolution2D(64, kernel_size=3, activation=activations.relu, padding="same")(inp)
    img_1 = Convolution2D(64, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPool2D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution2D(128, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution2D(128, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPool2D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution2D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution2D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool2D()(img_1)
    img_1 = Dropout(rate=0.1)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = PermaDropout(rate=0.1)(dense_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = PermaDropout(rate=0.1)(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    #model.summary()
    return model


if __name__ == "__main__":

    (X, Y), (X_test, Y_test) = mnist.load_data()

    X = X[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    X = X.astype(np.float)
    X_test = X_test.astype(np.float)

    print(X.shape)

    step_sizes = [128]*10

    unused_samples = list(range(X.shape[0]))
    step = np.random.choice(unused_samples, size=256).tolist()
    used_samples = step
    unused_samples = list(set(unused_samples) - set(step))
    val = np.random.choice(unused_samples, size=256).tolist()
    unused_samples = list(set(unused_samples) - set(val))

    results = []

    rnd = np.random.randint(1, 100000)

    for i, step_size in enumerate(step_sizes):
        X_used = X[used_samples, ...]
        Y_used = Y[used_samples, ...]

        X_val = X[val, ...]
        Y_val = Y[val, ...]

        X_unused = X[unused_samples, ...]
        Y_unused = Y[unused_samples, ...]

        model = get_model()

        model_name = "model.h5"

        check = ModelCheckpoint(filepath=model_name, monitor="val_acc", save_best_only=True, verbose=1,
                                save_weights_only=True)
        reduce = ReduceLROnPlateau(monitor="val_acc", patience=30, verbose=1, min_lr=1e-7)

        early = EarlyStopping(patience=40, monitor="val_acc")

        model.fit(X_used, Y_used, epochs=120, verbose=1, batch_size=32, validation_data=(X_val, Y_val),
                  callbacks=[check, reduce, early])

        model.load_weights(model_name)

        pred_test = model.predict(X_test, batch_size=1024)
        pred_test = np.argmax(pred_test, axis=-1)

        f1 = f1_score(Y_test, pred_test, average="macro")

        acc = accuracy_score(Y_test, pred_test)

        results.append({"size": X_used.shape[0], "accuracy": acc, "f1":f1})

        print(results[-1])

        step = np.random.choice(unused_samples, size=step_size).tolist()
        used_samples += step
        unused_samples = list(set(unused_samples) - set(step))
        os.makedirs('../output/mnist/', exist_ok=True)

        with open('../output/mnist/random_performance_%s.json'%float(rnd), 'w') as f:
            json.dump(results, f, indent=4)

        del model
        clear_session()

