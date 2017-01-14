import numpy as np
import mxnet as mx
import os
import math
import urllib
import gzip
import struct
import logging
import time

#import plot

# def read_data(rootDir):
# 	for lists in os.listdir(rootDir):
# 		path = os.path.join(rootDir, lists)
# 		if os.path.isdir(path):
# 			read_data(path)
# 		else
# 			a = np.readtxt(lists)

# 	return (label, tone)

# path1 = '/Users/SkyBiG/Desktop/Course/Deep Learning/Project/DL_Project/data-process-output/trim-interpolation/train'
# (train_lbl, train_tone) = read_data(path1)
# path2 = '/Users/SkyBiG/Desktop/Course/Deep Learning/Project/DL_Project/data-process-output/trim-interpolation/train'
# (val_lbl, val_tone) = read_data(path2)

################ READ_DATA
def read_data(label_url, tone_url):
	#with open(label_url, 'r') as f_lbl:
	label = np.loadtxt(label_url)
	#with open(tone_url, 'r') as f_tone:
	tone = np.loadtxt(tone_url) * 100
	#print label.shape
	print tone.shape
	return (label, tone)

#path = '/Users/SkyBiG/Desktop/Course/Deep Learning/Project/DL_Project/featured_data/'
path = '/Users/SkyBiG/Desktop/Course/Deep Learning/Project/DL/DL_Project/smooth_data/'
#path = '/Users/SkyBiG/Desktop/Course/Deep Learning/Project/DL_Project/data/'
#path = '/Users/SkyBiG/Desktop/Course/Deep Learning/Project/DL_Project/experiment/'
(train_lbl, train_tone) = read_data(
	#path + 'train_labels', path + 'featured_train_f0s')
	path + 'train_labels', path + 'train_f0s')
	#path + 'train_labels', path + 'train_f')
(val_lbl, val_tone) = read_data(
	#path + 'val_labels', path + 'featured_val_f0s')
	path + 'test_labels', path + 'test_f0s')
	#path + 'test_labels', path + 'test_f')
(test_lbl, test_tone) = read_data(
 	#path + 'test_labels', path + 'featured_test_f0s')
 	path + 'test_labels', path + 'test_f0s')
 	#path + 'test_labels', path + 'test_f')

################ FOR_ITER
def to4d(tone):
	#return tone.reshape(tone.shape[0] / 13, 1, 199, 13)
	#return tone.reshape(tone.shape[0], 1, 120, 1)
	return tone.reshape(tone.shape[0], 1, 120, 1)
	
batch_size = 20
train_iter = mx.io.NDArrayIter(to4d(train_tone), train_lbl, batch_size, shuffle = True)
val_iter = mx.io.NDArrayIter(to4d(val_tone), val_lbl, batch_size)
test_iter = mx.io.NDArrayIter(to4d(test_tone), test_lbl, batch_size)

################ MODEL

##### MLP
# Create a place holder variable for the input data
# data = mx.sym.Variable('data')

# conv1 = mx.sym.Convolution(data = data, kernel = (3, 1), num_filter = 32)
# act = mx.sym.Activation(data = conv1, name = 'relu', act_type = "relu")
# pool = mx.sym.Pooling(data = act, pool_type = "max", kernel = (2, 1), stride = (2, 1))

# conv2 = mx.sym.Convolution(data = pool, kernel = (3, 1), num_filter = 64)
# act_ = mx.sym.Activation(data = conv2, name = 'relu_', act_type = "relu")
# pool_ = mx.sym.Pooling(data = act_, pool_type = "max", kernel = (2, 1), stride = (2, 1))
# # Flatten the data from 4-D shape (batch_size, num_channel, width, height) 
# # into 2-D (batch_size, num_channel*width*height)
# data = mx.sym.Flatten(data = pool_)

# The first fully-connected layer
# fc1  = mx.sym.FullyConnected(data = data, name='fc1', num_hidden = 1024)
# # Apply relu to the output of the first fully-connnected layer
# act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

# # The second fully-connected layer and the according activation function
# fc2  = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 256)
# act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")

# # The thrid fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
# fc3  = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden = 4)
# # The softmax and loss layer
# net  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

##### LENET_change

data = mx.symbol.Variable('data')
# first conv layer
conv1 = mx.symbol.Convolution(data = data, kernel = (5, 1), num_filter = 20)
relu1 = mx.symbol.Activation(data = conv1, act_type = "relu")
pool1 = mx.symbol.Pooling(data = relu1, pool_type = "max", kernel = (2, 1) , stride = (2, 1))
# second conv layer
conv2 = mx.symbol.Convolution(data = pool1, kernel = (3, 1), num_filter = 50)
relu2 = mx.symbol.Activation(data = conv2, act_type = "relu")
pool2 = mx.symbol.Pooling(data = relu2, pool_type = "max", kernel = (2, 1), stride = (2, 1))

# conv3 = mx.sym.Convolution(data = pool2, kernel = (3, 1), num_filter = 64)
# relu3 = mx.sym.Activation(data = conv3, act_type = "relu")
# pool3 = mx.sym.Pooling(data = relu3, pool_type = "max", kernel = (2, 1), stride = (2, 1))

# first fullc layer
flatten = mx.symbol.Flatten(data = pool2)
fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 512)
relu1 = mx.symbol.Activation(data = fc1, act_type = "tanh")
# second fullc
# fc2 = mx.symbol.FullyConnected(data = tanh1, num_hidden = 512)
# tanh2 = mx.symbol.Activation(data = fc2, act_type = "tanh")

# fc2 = mx.symbol.FullyConnected(data = relu1, num_hidden = 42)
# relu2 = mx.symbol.Activation(data = fc2, act_type = "relu")

fc3 = mx.symbol.FullyConnected(data = relu1, num_hidden = 4)

# softmax loss
net = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

##### VGG

# data = mx.sym.Variable('data')

# conv1_1 = mx.sym.Convolution(data = data, kernel = (3, 1), num_filter = 16)
# relu1_1 = mx.sym.Activation(data = conv1_1, act_type = "relu")
# pool1 = mx.sym.Pooling(data = relu1_1, pool_type = "max", kernel = (2, 1), stride = (2, 1))

# conv2_1 = mx.sym.Convolution(data = pool1, kernel = (3, 1), num_filter = 32)
# relu2_1 = mx.sym.Activation(data = conv2_1, act_type = "relu")
# pool2 = mx.sym.Pooling(data = relu2_1, pool_type = "max", kernel = (2, 1), stride = (2, 1))

# conv3_1 = mx.sym.Convolution(data = pool2, kernel = (3, 1), num_filter = 64)
# relu3_1 = mx.sym.Activation(data = conv3_1, act_type = "relu")
# # conv3_2 = mx.sym.Convolution(data = relu3_1, kernel = (3, 1), num_filter = 8)
# # relu3_2 = mx.sym.Activation(data = conv3_2, act_type = "relu")
# pool3 = mx.sym.Pooling(data = relu3_1, pool_type = "max", kernel = (2, 1), stride = (2, 1))

# # conv4_1 = mx.sym.Convolution(data = pool3, kernel = (3, 1), num_filter = 64)
# # relu4_1 = mx.sym.Activation(data = conv4_1, act_type = "relu")
# # conv4_2 = mx.sym.Convolution(data = relu4_1, kernel = (3, 1), num_filter = 64)
# # relu4_2 = mx.sym.Activation(data = conv4_2, act_type = "relu")
# # pool4 = mx.sym.Pooling(data = relu4_2, pool_type = "max", kernel = (2, 1), stride = (2, 1))

# # conv5_1 = mx.sym.Convolution(data = pool4, kernel = (3, 1), num_filter = 64)
# # relu5_1 = mx.sym.Activation(data = conv5_1, act_type = "relu")
# # conv5_2 = mx.sym.Convolution(data = relu5_1, kernel = (3, 1), num_filter = 64)
# # relu5_2 = mx.sym.Activation(data = conv5_2, act_type = "relu")
# # pool5 = mx.sym.Pooling(data = relu5_2, pool_type = "max", kernel = (2, 1), stride = (2, 1))

# flatten = mx.sym.Flatten(data = pool3)
# fc6 = mx.sym.FullyConnected(data = flatten, num_hidden = 512)
# relu6 = mx.sym.Activation(data = fc6, act_type = "relu")
# #drop6 = mx.sym.Dropout(data = relu6, p = 0.5)

# fc7 = mx.sym.FullyConnected(data = relu6, num_hidden = 128)
# relu7 = mx.sym.Activation(data = fc7, act_type = "relu")
# #drop7 = mx.sym.Dropout(data = relu7, p = 0.5)

# fc8 = mx.sym.FullyConnected(data = relu7, num_hidden = 4)
# net = mx.sym.SoftmaxOutput(data = fc8, name = "softmax")


# model = mx.model.FeedForward(
#     symbol = net,       # network structure
#     num_epoch = 10,     # number of data passes for training 
#     learning_rate = 0.001 # learning rate of SGD 
# )

logging.basicConfig(level = logging.INFO)

model = mx.mod.Module(symbol = net)
#model_prefix = 'my_model'
st = time.time() 
#checkpoint = mx.callback.do_checkpoint(model_prefix)

model.fit(
    train_iter,       # training data
    eval_data = val_iter, # validation data
    optimizer = 'adam',
    optimizer_params = {'learning_rate':1e-4, 'decay_factor':0.95},
    num_epoch = 100,
    eval_metric = 'acc',
    #epoch_end_callback = checkpoint,
)

t = time.time() - st
print t
#model.predict(val_iter)
# for preds, i_batch, batch in model.iter_predict(data.get_iter(batch_size)):
#     pred_label = preds[0].asnumpy().argmax(axis = 1)
#     label = batch.label[0].asnumpy().astype('int32')
#     print('batch %d, accuracy %f' % (i_batch, float(sum(pred_label == label)) / len(label)))
##########################

test_acc = model.score(test_iter, eval_metric = 'acc')
print test_acc[0]





