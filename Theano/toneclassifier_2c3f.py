import theano
from theano import tensor as T
from theano.tensor.signal import pool
import numpy as np
from cnn_process_data import *

class ToneClassifier(object):
    def __init__(self, input_channels=None, input_columns=None, num_filters=None, filter_size=None, weight_scale=1e-2,
                 num_classes=None, hidden_size=None, reg=0, input_X=None, input_y=None):
        self.input = input_X
        self.y = input_y
        # CNN Layer 1
        self.cnn_weight_shape_0 = [num_filters[0], input_channels] + filter_size[0]
        self.conv_weights_0 = theano.shared(np.random.normal(0, weight_scale, self.cnn_weight_shape_0), name='conv_weight_0')
        self.cnn_bias_shape_0 = [num_filters[0]]
        self.conv_bias_0 = theano.shared(np.random.normal(1, weight_scale, self.cnn_bias_shape_0), name='conv_bias_0')
        self.conv_out_0 = T.nnet.relu(
            T.nnet.conv2d(self.input, self.conv_weights_0, border_mode='valid')
            + self.conv_bias_0.dimshuffle('x', 0, 'x', 'x'))
        self.pool_out_0 = pool.pool_2d(self.conv_out_0, (2, 1), ignore_border=False)

        # CNN Layer 2
        self.cnn_weight_shape_1 = [num_filters[1], num_filters[0]] + filter_size[1]
        self.conv_weights_1 = theano.shared(np.random.normal(0, weight_scale, self.cnn_weight_shape_1), name='conv_weight_1')
        self.cnn_bias_shape_1 = [num_filters[1]]
        self.conv_bias_1 = theano.shared(np.random.normal(1, weight_scale, self.cnn_bias_shape_1), name='conv_bias_1')
        self.conv_out_1 = T.nnet.relu(
            T.nnet.conv2d(self.pool_out_0, self.conv_weights_1, border_mode='valid')
            + self.conv_bias_1.dimshuffle('x', 0, 'x', 'x'))
        self.pool_out_1 = pool.pool_2d(self.conv_out_1, (2, 1), ignore_border=False)

        self.fully_connected_nn_input = self.pool_out_1.flatten(2)
        # FC layer 1
        self.affine_weights_0 = theano.shared(
            np.random.normal(0, weight_scale,
                             [num_filters[1] * 28, hidden_size[0]]),
            name="affine_weight_0")
        self.affine_bias_0 = theano.shared(
            np.random.normal(0, weight_scale, hidden_size[0]),
            name='affine_bias_0')
        self.hidden_output_0 = T.nnet.relu(
            T.dot(self.fully_connected_nn_input, self.affine_weights_0)
            + self.affine_bias_0.dimshuffle('x', 0))

        # FC layer 2
        self.affine_weights_1 = theano.shared(
            np.random.normal(0, weight_scale,
                             [hidden_size[0], hidden_size[1]]),
            name="affine_weight_1")
        self.affine_bias_1 = theano.shared(
            np.random.normal(0, weight_scale, hidden_size[1]),
            name="affine_bias_1")
        self.hidden_output_1 = T.nnet.relu(
            T.dot(self.hidden_output_0, self.affine_weights_1)
            + self.affine_bias_1.dimshuffle('x', 0))

        # Softmax Output Layer
        self.affine_weights_2 = theano.shared(
            np.random.normal(0, weight_scale,
                             [hidden_size[1], num_classes]),
            name="affine_weight_2")
        self.affine_bias_2 = theano.shared(
            np.random.normal(0, weight_scale, num_classes),
            name='affine_bias_2')
        self.softmax_out = T.nnet.softmax(
            T.dot(self.hidden_output_1, self.affine_weights_2)
            + self.affine_bias_2.dimshuffle('x', 0))

        self.y_pred = T.argmax(self.softmax_out, axis=1)
        self.error = T.mean(T.eq(self.y_pred, self.y))
        self.softmax_loss = -T.mean(T.log(self.softmax_out)[T.arange(self.y.shape[0]), self.y])

        self.reg = reg
        self.loss = self.softmax_loss \
                    + 0.5 * reg * (T.sum(self.conv_weights_0) ** 2
                                   + T.sum(self.conv_weights_1) ** 2
                                   + T.sum(self.affine_weights_0) ** 2
                                   + T.sum(self.affine_weights_1) ** 2
                                   + T.sum(self.affine_weights_2) ** 2)

        self.params = [self.conv_weights_0, self.conv_bias_0,
                       self.conv_weights_1, self.conv_bias_1,
                       self.affine_weights_0, self.affine_bias_0,
                       self.affine_weights_1, self.affine_bias_1,
                       self.affine_weights_2, self.affine_bias_2]
