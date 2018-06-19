import keras
import numpy as np
import keras.backend as K


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return np.dot(features, K.transpose(features))


def style_loss(style, combination):
    s_gram = gram_matrix(style)
    c_gram = gram_matrix(combination)
    channels = 3
    ht, wdth = style.shape
    np.sum(np.square(s_gram-c_gram)) / (4*(channels**2)*(ht*wdth)**2)


def content_loss(content, combination):
    return K.sum(K.square(combination-content))


def variation_loss(x):
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
