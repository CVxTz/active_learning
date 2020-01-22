from glob import glob
from random import sample

import imgaug.augmenters as iaa
import numpy as np

from image_utils import read_img, select_random_crop, resize_img
from learn_embedding import embedding_model
import json
from tqdm import tqdm


img_size = 128


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


if __name__ == '__main__':

    model_name = "embedding_model.h5"
    batch_size = 16

    print("Model : %s" % model_name)

    paths = list(sorted(glob('/media/ml/data_ml/dogs-vs-cats-redux-kernels-edition/train/*.jpg')))

    model = embedding_model()

    model.load_weights(model_name, by_name=True)

    all_pred = []

    for batch_paths in tqdm(chunker(paths, size=16), total=len(paths)//16):
        batch_images = [read_img(x) for x in batch_paths]
        batch_images = [resize_img(x, h=2 * img_size, w=2 * img_size) for x in batch_images]

        X = np.array(batch_images).astype(float)

        all_pred += model.predict(X).tolist()

    _output = {k.split('/')[-1]: v for k, v in zip(paths, all_pred)}

    json.dump(_output, open('../output/img_embeddings.json', 'w'), indent=4)
