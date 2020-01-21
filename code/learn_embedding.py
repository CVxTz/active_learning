from glob import glob
from random import sample

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool2D, Multiply, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import imgaug.augmenters as iaa
from image_utils import read_img, select_random_crop, resize_img

img_size = 128


def baseline_model():
    input_1 = Input(shape=(None, None, 3))
    input_2 = Input(shape=(None, None, 3))

    base_model = ResNet50(weights="imagenet", include_top=False)

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    x1 = GlobalMaxPool2D()(x1)
    x2 = GlobalMaxPool2D()(x2)

    D = Dense(50, activation='selu', name="embed")
    BN = BatchNormalization(name="bn")

    x1 = D(x1)
    x2 = D(x2)

    x1 = BN(x1)
    x2 = BN(x2)

    x = Multiply()([x1, x2])

    x = Dropout(0.1)(x)

    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=Adam(0.00001))

    model.summary()

    return model


def embedding_model():
    input_1 = Input(shape=(None, None, 3))

    base_model = ResNet50(weights="imagenet", include_top=False)

    x1 = base_model(input_1)

    x1 = GlobalMaxPool2D()(x1)

    D = Dense(50, activation='selu', name="embed")
    BN = BatchNormalization(name="bn")

    x1 = D(x1)
    x1 = BN(x1)

    model = Model(input_1, x1)

    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=Adam(0.00001))

    model.summary()

    return model


def gen(list_paths, batch_size=16):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 3.0)),
    ])
    while True:
        batch_paths = sample(list_paths, batch_size // 2)
        batch_images = [read_img(x) for x in batch_paths]

        X1 = [select_random_crop(x) for x in batch_images]
        X2 = [select_random_crop(x) for x in batch_images]
        labels = [1] * len(batch_images)

        X1 += [select_random_crop(x) for x in batch_images]
        X2 += [select_random_crop(x) for x in batch_images[::-1]]
        labels += [0] * len(batch_images)

        X1 = seq(images=X1)
        X2 = seq(images=X2)

        X1 = [resize_img(x, h=img_size, w=img_size) for x in X1]
        X2 = [resize_img(x, h=img_size, w=img_size) for x in X2]

        X1 = np.array(X1)
        X2 = np.array(X2)

        labels = np.array(labels)

        yield [X1, X2], labels


if __name__ == '__main__':

    model_name = "embedding_model.h5"
    batch_size = 16

    print("Model : %s" % model_name)

    paths = list(sorted(glob('/media/ml/data_ml/dogs-vs-cats-redux-kernels-edition/train/*.jpg')))
    train, test = train_test_split(paths, random_state=1337, test_size=0.1)

    train, val = train_test_split(train, random_state=1337, test_size=0.1)

    model = baseline_model()

    try:
        model.load_weights(model_name, by_name=True)
    except:
        pass

    check = ModelCheckpoint(filepath=model_name, monitor="val_accuracy", save_best_only=True, verbose=1,
                            save_weights_only=True)
    reduce = ReduceLROnPlateau(monitor="val_accuracy", patience=50, verbose=1, min_lr=1e-7)

    history = model.fit_generator(gen(train, batch_size=batch_size), epochs=10000, verbose=1,
                                  steps_per_epoch=len(train) // batch_size // 10,
                                  validation_data=gen(val, batch_size=batch_size),
                                  validation_steps=len(val) // batch_size // 10,
                                  use_multiprocessing=True, workers=8, callbacks=[check, reduce])
