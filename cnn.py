import process_data
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano_cnn_1c1d import conv1d_multi_channel_single_row
import numpy as np

engy_train, f0_train, y_train = process_data.LoadAndProcessTrainData()
engy_val, f0_val, y_val = process_data.LoadAndProcessValData()

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
W = theano.shared(np.random.normal(0, weight_scale, cnn_weight_shape), name='W')
cnn_bias_shape = [filter_num]
b = theano.shared(np.random.normal(1, weight_scale, cnn_bias_shape), name='b')

# build symbolic expression that computes the convolution of input with filters in w

conv_out = T.nnet.relu(conv1d_multi_channel_single_row(input, W, border_mode='half') + b.dimshuffle('x', 0, 'x'))

