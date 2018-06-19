import keras
import numpy as np
from keras.applications import vgg19
from keras.preprocessing.image import load_img
from scipy.optimize import fmin_l_bfgs_b
from losses import content_loss, style_loss, variation_loss
from utils import preprocess_img, decompress_img


class Neural_sytler:

    def __init__(self, content_img_path, style_img_path, output_img_path, output_width, content_weight, style_weight, tv_weight, content_layer, style_layers, iterations):

        self.content_img_path = content_img_path
        self.style_img_path = style_img_path
        self.output_img_path = output_img_path

        print('\n\tResizing images...')
        self.width = output_width
        width, height = load_img(self.content_img_path).size
        new_dims = (height, width)
        self.img_nrows = height
        self.img_ncols = width

        if self.width is not None:
            num_rows = int(np.floor(float(height * self.width / width)))
            new_dims = (num_rows, self.width)
            self.img_nrows = num_rows
            self.img_ncols = self.width

        self.content_img = K.variable(preprocess_image(self.content_img_path, new_dims))
        self.style_img = K.variable(preprocess_image(self.style_img_path, new_dims))
        self.combination_img = K.placeholder((1, new_dims[0], new_dims[1], 3))

        print("\tSize of content image is: {}".format(K.int_shape(self.content_img)))
        print("\tSize of style image is: {}".format(K.int_shape(self.style_img)))
        print("\tSize of output image is: {}".format(K.int_shape(self.output_img)))

        self.input_img = K.concatenate([self.content_img, self.style_img, self.output_img], axis=0)
        self.iterations = iterations
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.content_layer = content_layer
        self.style_layers = style_layers

        self.model = vgg19.VGG19(input_tensor=self.input_img, weights='imagenet', include_top=False)
        print('Model loaded...')

        print('\tComputing losses...')
        outputs_dict = dict([(layer.name, layer.output) for layer in self.model.layers])

        content_features = outputs_dict[self.content_layer]

        base_image_features = content_features[0, :, :, :] 	# 0 corresponds to base
        combination_features = content_features[2, :, :, :]  # 2 coresponds to output

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
            """
            Run L-BFGS over the pixels of the generated image so as to
            minimize the neural style loss.
            """

        if K.image_dim_ordering() == 'th':
            x = np.random.uniform(0, 255, (1, 3, self.img_nrows, self.img_ncols)) - 128.
        else:
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
        if K.image_dim_ordering() == 'th':
            x = x.reshape((1, 3, self.img_nrows, self.img_ncols))
        else:
            x = x.reshape((1, self.img_nrows, self.img_ncols, 3))

        outs = self.loss_and_grads([x])
        loss_value = outs[0]
        return loss_value

    def grads(self, x):
        if K.image_dim_ordering() == 'th':
            x = x.reshape((1, 3, self.img_nrows, self.img_ncols))
        else:
            x = x.reshape((1, self.img_nrows, self.img_ncols, 3))

        outs = self.loss_and_grads([x])

        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return grad_values
