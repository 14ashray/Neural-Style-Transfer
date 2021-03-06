from __future__ import print_function

import time
import numpy as np
from scipy.misc import imsave
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from keras.applications import vgg16, vgg19
from keras.preprocessing.image import load_img
from utils import preprocess_image, deprocess_image
from losses import style_reconstruction_loss, feature_reconstruction_loss, total_variation_loss


class Neural_Styler(object):
    def __init__(self,
                 base_img_path,
                 style_img_path,
                 output_img_path,
                 output_width,
                 convnet,
                 content_weight,
                 style_weight,
                 tv_weight,
                 content_layer,
                 style_layers,
                 iterations):

        self.base_img_path = base_img_path
        self.style_img_path = style_img_path
        self.output_img_path = output_img_path

        self.width = output_width
        width, height = load_img(self.base_img_path).size
        new_dims = (height, width)

        self.img_nrows = height
        self.img_ncols = width

        if self.width is not None:
            num_rows = int(np.floor(float(height * self.width / width)))
            new_dims = (num_rows, self.width)

            self.img_nrows = num_rows
            self.img_ncols = self.width

        self.content_img = K.variable(preprocess_image(self.base_img_path, new_dims))
        self.style_img = K.variable(preprocess_image(self.style_img_path, new_dims))

        if K.image_dim_ordering() == 'th':
            self.output_img = K.placeholder((1, 3, new_dims[0], new_dims[1]))
        else:
            self.output_img = K.placeholder((1, new_dims[0], new_dims[1], 3))

        print("\tSize of content image is: {}".format(K.int_shape(self.content_img)))
        print("\tSize of style image is: {}".format(K.int_shape(self.style_img)))
        print("\tSize of output image is: {}".format(K.int_shape(self.output_img)))

        self.input_img = K.concatenate([self.content_img,
                                        self.style_img,
                                        self.output_img], axis=0)

        self.convnet = convnet
        self.iterations = iterations

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

        self.content_layer = content_layer
        self.style_layers = style_layers

        print('\tLoading {} model'.format(self.convnet.upper()))

        if self.convnet == 'vgg16':
            self.model = vgg16.VGG16(input_tensor=self.input_img,
                                     weights='imagenet',
                                     include_top=False)
        else:
            self.model = vgg19.VGG19(input_tensor=self.input_img,
                                     weights='imagenet',
                                     include_top=False)

        outputs_dict = dict([(layer.name, layer.output) for layer in self.model.layers])
        content_features = outputs_dict[self.content_layer]

        base_image_features = content_features[0, :, :, :]
        combination_features = content_features[2, :, :, :]

        content_loss = self.content_weight * \
            feature_reconstruction_loss(base_image_features,
                                        combination_features)

        temp_style_loss = K.variable(0.0)
        weight = 1.0 / float(len(self.style_layers))

        for layer in self.style_layers:
            style_features = outputs_dict[layer]
            style_image_features = style_features[1, :, :, :]
            output_style_features = style_features[2, :, :, :]
            temp_style_loss += weight * \
                style_reconstruction_loss(style_image_features,
                                          output_style_features,
                                          self.img_nrows,
                                          self.img_ncols)
        style_loss = self.style_weight * temp_style_loss

        tv_loss = self.tv_weight * total_variation_loss(self.output_img,
                                                        self.img_nrows,
                                                        self.img_ncols)

        total_loss = content_loss + style_loss + tv_loss

        print('\tComputing gradients...')
        grads = K.gradients(total_loss, self.output_img)

        outputs = [total_loss]
        if type(grads) in {list, tuple}:
            outputs += grads
        else:
            outputs.append(grads)

        self.loss_and_grads = K.function([self.output_img], outputs)

    def style(self):
        x = np.random.uniform(0, 255, (1, self.img_nrows, self.img_ncols, 3)) - 128.

        for i in range(self.iterations):
            print('\n\tIteration: {}'.format(i+1))

            toc = time.time()
            x, min_val, info = fmin_l_bfgs_b(self.loss, x.flatten(), fprime=self.grads, maxfun=20)

            img = deprocess_image(x.copy(), self.img_nrows, self.img_ncols)
            fname = self.output_img_path + '_at_iteration_%d.png' % (i+1)
            imsave(fname, img)

            tic = time.time()

            print('\t\tImage saved as', fname)
            print('\t\tLoss: {:.2e}, Time: {} seconds'.format(float(min_val), float(tic-toc)))

    def loss(self, x):
        x = x.reshape((1, self.img_nrows, self.img_ncols, 3))

        outs = self.loss_and_grads([x])
        loss_value = outs[0]
        return loss_value

    def grads(self, x):
        x = x.reshape((1, self.img_nrows, self.img_ncols, 3))

        outs = self.loss_and_grads([x])

        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return grad_values
