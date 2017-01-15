import timeit
import theano
import theano.tensor as T
import numpy as np
from cnn_process_data import *
from toneclassifier_2c2f import ToneClassifier
import data_utils
import cnn_utils

data_utils.SetPath('../toneclassifier')

input_columns = 120
num_classes = 4
batch_size = 20

X_train, y_train = ImportData(input_columns, 'train')
X_train_shared, y_train_shared = theano.shared(X_train), theano.shared(y_train)
X_val, y_val = ImportData(input_columns, 'val')
X_val_shared, y_val_shared = theano.shared(X_val), theano.shared(y_val)
X_test, y_test = ImportData(input_columns, 'test')
X_test_shared, y_test_shared = theano.shared(X_test), theano.shared(y_test)

n_train_batches = X_train.shape[0] / batch_size
n_valid_batches = X_val.shape[0] / batch_size
n_test_batches = X_test.shape[0] / batch_size

learning_rate = 5e-4

eta = theano.shared(np.array(learning_rate, dtype=theano.config.floatX))
eta_decay = np.array(0.95, dtype=theano.config.floatX)

index = T.lscalar()  # index to a [mini]batch
X = T.tensor4('X')  # the data is presented as rasterized images
y = T.ivector('y')

# toneclassifier_2c3f
# toneclassifer = ToneClassifier(
#     input_channels=2, input_columns=input_columns,
#     num_filters=[32, 64], filter_size=[[5, 1], [3, 1]],
#     hidden_size=[512, 128], num_classes=num_classes, input_X=X, input_y=y, reg=1e-4)

# toneclassisifier_2c2f
toneclassifer = ToneClassifier(
    input_channels=1, input_columns=input_columns,
    num_filters=[20, 80], filter_size=[[5, 1], [3, 1]],
    hidden_size=640, num_classes=num_classes, input_X=X, input_y=y, reg=0, weight_scale=5e-3)


# d_params = [
#     T.grad(toneclassifer.loss, param) for param in toneclassifer.params
# ]
#
# updates = [
#     (param, param - eta * d_param) for param, d_param in zip(toneclassifer.params, d_params)
# ]

updates = cnn_utils.adam(loss=toneclassifer.loss,
                         all_params=toneclassifer.params,
                         learning_rate=learning_rate)

train_model = theano.function(
    inputs=[index],
    outputs=toneclassifer.loss,
    updates=updates,
    givens={
        X: X_train_shared[index * batch_size: (index + 1) * batch_size],
        y: y_train_shared[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    inputs=[index],
    outputs=[toneclassifer.error, toneclassifer.loss],
    givens={
        X: X_val_shared[index * batch_size: (index + 1) * batch_size],
        y: y_val_shared[index * batch_size: (index + 1) * batch_size]
    }
)

test_model = theano.function(
    inputs=[index],
    outputs=toneclassifer.error,
    givens={
        X: X_test_shared[index * batch_size: (index + 1) * batch_size],
        y: y_test_shared[index * batch_size: (index + 1) * batch_size]
    }
)

validation_frequency = n_train_batches
                              # go through this many
                              # minibatche before checking the network
                              # on the validation set; in this case we
                              # check every epoch

best_validation_loss = np.inf
best_train_loss = np.inf
best_train_acc = 0.
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()
epoch = 0
done_looping = False
n_epochs = 300

while (epoch < n_epochs) and (not done_looping):
    X_train, y_train = data_utils.unison_shuffled_copies(X_train, y_train)
    X_val, y_val = data_utils.unison_shuffled_copies(X_val, y_val)
    X_test, y_test = data_utils.unison_shuffled_copies(X_test, y_test)
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):

        minibatch_avg_cost = train_model(minibatch_index)
        # iteration number
        iter = (epoch - 1) * n_train_batches + minibatch_index

        if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validataions = [validate_model(i) for i in range(n_valid_batches)]
            validation_acc = [validation[0] for validation in validataions]
            validation_loss = [validation[1] for validataion in validataions]
            this_validation_acc = np.mean(validation_acc)
            this_validation_loss = np.mean(validation_loss)
            print(
                'epoch %i, minibatch %i/%i, train loss %f, validation loss %f, validation accuracy %f %%' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    minibatch_avg_cost,
                    this_validation_loss,
                    this_validation_acc * 100.
                )
            )
            if this_validation_loss < best_validation_loss or this_validation_acc > best_train_acc:
                best_validation_loss = this_validation_loss
                best_train_acc = this_validation_acc
                best_iter = iter

                # test it on the test set
                test_losses = [test_model(i) for i
                               in range(n_test_batches)]
                test_score = np.mean(test_losses)

                print(('     epoch %i, minibatch %i/%i, test accuracy of '
                       'best model %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score * 100.))

    eta.set_value(eta.get_value() * eta_decay)
end_time = timeit.default_timer()
print(('Optimization complete. Best validation score of %f %% '
       'obtained at iteration %i, with test performance %f %%') %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.))

test_losses = [test_model(i) for i in range(n_test_batches)]
test_score = np.mean(test_losses)
print(('final test accuracy of best model %f %%') % (test_score * 100.))