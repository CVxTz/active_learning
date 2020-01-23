import json
from random import sample

import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from image_utils import read_img, resize_img

img_size = 128


def baseline_model():
    input_1 = Input(shape=(None, None, 3))

    base_model = ResNet50(weights="imagenet", include_top=False)

    x1 = base_model(input_1)

    x1 = GlobalMaxPool2D()(x1)

    D = Dense(50, activation='selu', name="embed")
    BN = BatchNormalization(name="bn")

    x1 = D(x1)
    x1 = BN(x1)

    x = Dense(1, activation="sigmoid")(x1)

    model = Model(input_1, x)

    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=Adam(0.00001))

    model.summary()

    return model


def gen(list_paths, batch_size=16):
    while True:
        batch_paths = sample(list_paths, batch_size)
        batch_images = [read_img(x) for x in batch_paths]

        labels = [1 if "cat" in x.split("/")[-1] else 0 for x in batch_paths]

        X1 = [resize_img(x, h=2 * img_size, w=2 * img_size) for x in batch_images]

        X1 = np.array(X1)

        labels = np.array(labels)

        yield X1, labels


if __name__ == '__main__':

    model_name = "random_baseline.h5"
    pre_trained_name = "embedding_model.h5"

    batch_size = 16

    print("Model : %s" % model_name)

    with open('../output/learning_steps.json', 'r') as f:
        data = json.load(f)
        train, test, val = data['train'], data['test'], data['val']

        #active_learning_steps = data['active_learning_steps']
        random_steps = data['random_steps']

    used_train = []

    perf_curve = []

    model = baseline_model()

    try:
        # model.load_weights(model_name, by_name=True)
        model.load_weights(pre_trained_name, by_name=True)
    except:
        pass

    for step in random_steps:

        used_train += step


        check = ModelCheckpoint(filepath=model_name, monitor="val_accuracy", save_best_only=True, verbose=1,
                                save_weights_only=True)
        reduce = ReduceLROnPlateau(monitor="val_accuracy", patience=5, verbose=1, min_lr=1e-7)

        early = EarlyStopping(patience=10)

        history = model.fit_generator(gen(used_train, batch_size=batch_size), epochs=30, verbose=1,
                                      steps_per_epoch=len(used_train) // batch_size,
                                      validation_data=gen(val, batch_size=batch_size),
                                      validation_steps=len(val) // batch_size,
                                      use_multiprocessing=True, workers=8, callbacks=[check, reduce])

        model.load_weights(model_name)

        perf = model.evaluate_generator(gen(test, batch_size=batch_size), steps=2*len(test) // batch_size,
                                        use_multiprocessing=True, workers=8)

        d = {"accuracy": float(perf[1]), "size": len(used_train)}

        perf_curve.append(d)

        with open('../output/random_performance.json', 'w') as f:

            json.dump(perf_curve, f, indent=4)


