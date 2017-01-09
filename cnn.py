import process_data
import theano
from theano import tensor as T
from theano_cnn_1c1d import conv1d_multi_channel_single_row
import numpy as np

input_columns = 200
num_classes = 4

engy_train, f0_train, y_train = process_data.LoadAndProcessTrainData(input_columns)
engy_val, f0_val, y_val = process_data.LoadAndProcessValData(input_columns)

X_train = [engy_train, f0_train]
X_train = np.asarray(X_train)
X_train = X_train.swapaxes(0, 1)

X_val = [engy_val, f0_val]
X_val = np.asarray(X_val)
X_val = X_val.swapaxes(0, 1)


# instantiate 4D tensor for input
input = T.tensor3(name='input')

# initialize shared variable for weights.
input_channel = 2
filter_num = 16
filter_size = 5
weight_scale = 1e-3

# cnn weight shape=[output_channels, input_channels, output_columns]
# output_channels = filter_num
# input_channels = 2
cnn_weight_shape = [filter_num, input_channel, filter_size]
conv_weights = theano.shared(np.random.normal(0, weight_scale, cnn_weight_shape), name='conv_weight')
cnn_bias_shape = [filter_num]
conv_bias = theano.shared(np.random.normal(1, weight_scale, cnn_bias_shape), name='conv_bias')

# build symbolic expression that computes the convolution of input with filters in w

conv_out = T.nnet.relu(conv1d_multi_channel_single_row(input, conv_weights, border_mode='half') + conv_bias.dimshuffle('x', 0, 'x'))

fully_connected_nn_input = conv_out.flatten(2)
hidden_size_1 = 100
affine_weights_1 = theano.shared(
    np.random.normal(0, weight_scale,
                     [filter_num * input_columns, hidden_size_1]),
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
softmax_loss = -T.mean(T.log(softmax_out)[T.arange(y.shape[0]), y])