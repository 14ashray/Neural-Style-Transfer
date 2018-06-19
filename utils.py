import numpy as np
import kears
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array


def preprocess_img(img_path):
    img = load_img(img_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_img(img)
    return img


def decompress_img(x):
    img = x.reshape((img_rows, img_cols, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
