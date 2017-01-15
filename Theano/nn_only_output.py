import theano
from theano import tensor as T
from theano.tensor.signal import pool
import numpy as np
from cnn_process_data import *

class ToneClassifier(object):
    def __init__(self, input_channels=None, input_columns=None, weight_scale=2e-2,
                 num_classes=None, reg=0, input_X=None, input_y=None):
        self.input = input_X
        self.y = input_y
        # CNN Layer 1

        # FC layer 1
        # Softmax Output Layer
        self.affine_weights_2 = theano.shared(
            np.random.normal(0, weight_scale,
                             [input_columns, num_classes]),
            name="affine_weight_2")
        self.affine_bias_2 = theano.shared(
            np.random.normal(0, weight_scale, num_classes),
            name='affine_bias_2')
        self.softmax_out = T.nnet.softmax(
            T.dot(self.input.flatten(2), self.affine_weights_2)
            + self.affine_bias_2.dimshuffle('x', 0))

        self.y_pred = T.argmax(self.softmax_out, axis=1)
        self.error = T.mean(T.eq(self.y_pred, self.y))
        self.softmax_loss = -T.mean(T.log(self.softmax_out)[T.arange(self.y.shape[0]), self.y])

        self.reg = reg
        self.loss = self.softmax_loss \
                    + 0.5 * reg * (T.sum(self.affine_weights_2) ** 2)

        self.params = [self.affine_weights_2, self.affine_bias_2]
