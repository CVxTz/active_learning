import numpy as np
import cv2
import os
from tqdm import tqdm
from glob import glob
from pathlib import Path
from random import randint


def read_img(path):
    img = cv2.imread(path)
    return img


def resize_img(img, h=128, w=128):

    desired_size_h = h
    desired_size_w = w

    old_size = img.shape[:2]

    ratio = min(desired_size_w/old_size[1], desired_size_h/old_size[0])

    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size_w - new_size[1]
    delta_h = desired_size_h - new_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im


def crop(img, crop_size_x, crop_size_y):
    imgheight = img.shape[0]
    imgwidth = img.shape[1]
    i = randint(0, imgheight - crop_size_x)
    j = randint(0, imgwidth - crop_size_y)

    return img[i:(i + crop_size_x), j:(j + crop_size_y), :]


def select_random_crop(img):
    ratio_x = np.random.uniform(0.3, 0.6)
    ratio_y = np.random.uniform(0.3, 0.6)

    crop_size_x, crop_size_y = int(img.shape[0]*ratio_x), int(img.shape[1]*ratio_y)

    img = crop(img, crop_size_x, crop_size_y)

    return img


if __name__ == '__main__':

    img = cv2.imread("/media/ml/data_ml/dogs-vs-cats-redux-kernels-edition/train/cat.20.jpg")

    img = select_random_crop(img)

    img = resize_img(img, h=256, w=256)

    cv2.imwrite("sample_2.jpg", img)