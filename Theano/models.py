import theano
from theano import tensor as T
from theano_cnn_1c1d import conv1d_multi_channel_single_row
import numpy as np
from cnn_process_data import *

input_columns = 200
num_classes = 4

X_train, y_train = ImportData(input_columns, 'train')
X_val, y_val = ImportData(input_columns, 'val')
X_test, y_test = ImportData(input_columns, 'test')

# instantiate 4D tensor for input
network_input = T.tensor3(name='network_input')

# initialize shared variable for weights.
input_channel = 2
num_filters = 16
filter_size = 5
weight_scale = 1e-3


class Classifiers(object):
    def __init__(self, input_channels=None, input_columns=None, num_filters=None, filter_size=None, weight_scale=1e-3,
                 num_classes=None, hidden_layers=None, reg=1e-4):
        cnn_weight_shape = [num_filters, input_channel, filter_size]
        conv_weights = theano.shared(np.random.normal(0, weight_scale, cnn_weight_shape), name='conv_weight')
        cnn_bias_shape = [num_filters]
        conv_bias = theano.shared(np.random.normal(1, weight_scale, cnn_bias_shape), name='conv_bias')
        conv_out = T.nnet.relu(
            conv1d_multi_channel_single_row(network_input,
                                            conv_weights,
                                            border_mode='half')
            + conv_bias.dimshuffle('x', 0, 'x'))
        fully_connected_nn_input = conv_out.flatten(2)
        affine_weights_1 = theano.shared(
            np.random.normal(0, weight_scale,
                             [num_filters * input_columns, hidden_size_1]),
            name="affine_weight_1")


# build symbolic expression that computes the convolution of input with filters in w

hidden_size_1 = 100
affine_weights_1 = theano.shared(
    np.random.normal(0, weight_scale,
                     [num_filters * input_columns, hidden_size_1]),
    name="affine_weight_1")
affine_bias_1 = theano.shared(
    np.random.normal(0, weight_scale, hidden_size_1),
    name='affine_bias_1')
hidden_output_1 = T.nnet.relu(T.dot(fully_connected_nn_input, affine_weights_1) + affine_bias_1.dimshuffle('x', 0))

hidden_size_2 = num_classes
affine_weights_2 = theano.shared(
    np.random.normal(0, weight_scale,
                     [hidden_size_1, hidden_size_2]),
    name="affine_weight_2")
affine_bias_2 = theano.shared(
    np.random.normal(0, weight_scale, hidden_size_2),
    name='affine_bias_2')
softmax_out = T.nnet.softmax(T.dot(hidden_output_1, affine_weights_2) + affine_bias_2.dimshuffle('x', 0))

y = T.ivector(name='y')
y_pred = T.argmax(softmax_out, axis=1)
error_y = T.mean(T.neq(y_pred, y))
softmax_loss = -T.mean(T.log(softmax_out)[T.arange(y.shape[0]), y])

f = theano.function(inputs=[network_input, y],
                    outputs=[softmax_loss, error_y])
print f(X_train, y_train)
