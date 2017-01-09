import process_data
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano_cnn_1c1d import conv1d_multi_channel_single_row
import numpy

engy_train, f0_train, y_train = process_data.LoadAndProcessTrainData()
engy_val, f0_val, y_val = process_data.LoadAndProcessValData()

rng = numpy.random.RandomState(23455)

# instantiate 4D tensor for input
input = T.tensor3(name='input')

# initialize shared variable for weights.
w_shp = (16, 2, 5)
w_bound = numpy.sqrt(2 * 1 * 200)
W = theano.shared( numpy.asarray(
            rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_shp),
            dtype=input.dtype), name ='W')

# initialize shared variable for bias (1D tensor) with random values
# IMPORTANT: biases are usually initialized to zero. However in this
# particular application, we simply apply the convolutional layer to
# an image without learning the parameters. We therefore initialize
# them to random values to "simulate" learning.
b_shp = (16,)
b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input.dtype), name ='b')

# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv1d_multi_channel_single_row(input, W, border_mode='half')

# build symbolic expression to add bias and apply activation function, i.e. produce neural net layer output
# A few words on ``dimshuffle`` :
#   ``dimshuffle`` is a powerful tool in reshaping a tensor;
#   what it allows you to do is to shuffle dimension around
#   but also to insert new ones along which the tensor will be
#   broadcastable;
#   dimshuffle('x', 2, 'x', 0, 1)
#   This will work on 3d tensors with no broadcastable
#   dimensions. The first dimension will be broadcastable,
#   then we will have the third dimension of the input tensor as
#   the second of the resulting tensor, etc. If the tensor has
#   shape (20, 30, 40), the resulting tensor will have dimensions
#   (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor)
#   More examples:
#    dimshuffle('x') -> make a 0d (scalar) into a 1d vector
#    dimshuffle(0, 1) -> identity
#    dimshuffle(1, 0) -> inverts the first and second dimensions
#    dimshuffle('x', 0) -> make a row out of a 1d vector (N to 1xN)
#    dimshuffle(0, 'x') -> make a column out of a 1d vector (N to Nx1)
#    dimshuffle(2, 0, 1) -> AxBxC to CxAxB
#    dimshuffle(0, 'x', 1) -> AxB to Ax1xB
#    dimshuffle(1, 'x', 0) -> AxB to Bx1xA
output = T.nnet.relu(conv_out + b.dimshuffle('x', 0, 'x'))

# create theano function to compute filtered images
f = theano.function([input], output)

a_data = [engy_train[0], engy_val[0]]
a_data = numpy.asarray(a_data)
a_data_ = a_data.reshape([1, a_data.shape[0], a_data.shape[1]])
print a_data_.shape

filtered_data = f(a_data_)
print filtered_data.shape
