#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

class Worker:

    def load_dataset(self):
        # We first define a download function, supporting both Python 2 and 3.
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)

        # We then define functions for loading MNIST images and labels.
        # For convenience, they also download the requested files if needed.
        import gzip

        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the inputs in Yann LeCun's binary format.
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            # The inputs are vectors now, we reshape them to monochrome 2D images,
            # following the shape convention: (examples, channels, rows, columns)
            data = data.reshape(-1, 1, 28, 28)
            # The inputs come as bytes, we convert them to float32 in range [0,1].
            # (Actually to range [0, 255/256], for compatibility to the version
            # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
            return data / np.float32(256)

        def load_mnist_labels(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the labels in Yann LeCun's binary format.
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            # The labels are vectors of integers now, that's exactly what we want.
            return data

        # We can now download and read the training and test set images and labels.
        X_train = load_mnist_images('train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
        X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
        y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

        # We reserve the last 10000 training examples for validation.
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def build_cnn(self, input_var=None, nfilters = 32):
        # As a third model, we'll create a CNN of two convolution + pooling stages
        # and a fully-connected hidden layer in front of the output layer.

        # Input layer, as usual:
        network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                            input_var=input_var)
        # This time we do not apply input dropout, as it tends to work less well
        # for convolutional layers.

        # Convolutional layer with 32 kernels of size 5x5. Strided and padded
        # convolutions are supported as well; see the docstring.
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=nfilters, filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        # Expert note: Lasagne provides alternative convolutional layers that
        # override Theano's choice of which implementation to use; for details
        # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

        # Max-pooling layer of factor 2 in both dimensions:
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=nfilters, filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        # A fully-connected layer of 256 units with 50% dropout on its inputs:
        network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=256,
                nonlinearity=lasagne.nonlinearities.rectify)

        # And, finally, the 10-unit output layer with 50% dropout on its inputs:
        network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax)

        return network

    # ############################# Batch iterator ###############################
    # This is just a simple helper function iterating over training data in
    # mini-batches of a particular size, optionally in random order. It assumes
    # data is available as numpy arrays. For big datasets, you could load numpy
    # arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
    # own custom data iteration function. For small datasets, you can also copy
    # them to GPU at once for slightly improved performance. This would involve
    # several changes in the main program, though, and is not demonstrated here.
    # Notice that this function returns only mini-batches of size `batchsize`.
    # If the size of the data is not a multiple of `batchsize`, it will not
    # return the last (remaining) mini-batch.

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    # ############################## Main program ################################
    # Everything else will be handled in our main program now. We could pull out
    # more functions to better separate the code, but it wouldn't make it any
    # easier to read.

    def get_result(self, ntrain = 50000, nvalid = 10000, ntest = 10000, algorithm_type = 1,
            batch_size_train = 500, batch_size_valid = 500, batch_size_test = 500,
            num_epochs=500, stat_filename = 'stat.txt', LR = 0.1, M = 0.9,
            nfilters = 32, time_limit = 10000):
        # Load the dataset
        print("Loading data...")
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_dataset()

        X_train = X_train[1:ntrain]
        y_train = y_train[1:ntrain]
        X_val = X_val[1:nvalid]
        y_val = y_val[1:nvalid]
        X_test = X_test[1:ntest]
        y_test = y_test[1:ntest]

        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        # Create neural network model (depending on first command line parameter)
        print("Building model and compiling functions...")
        network = self.build_cnn(input_var, nfilters)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(network, trainable=True)
        if (algorithm_type == 1):
            updates = lasagne.updates.sgd(loss, params, learning_rate=LR)
        if (algorithm_type == 2):
            updates = lasagne.updates.momentum(loss, params, learning_rate=LR,
                                            momentum = M)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                        dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        # Finally, launch the training loop.
        nparameters = lasagne.layers.count_params(network, trainable=True)
        print("Number of parameters in model: {}".format(nparameters))
        print("Starting training...")

        stat_file = open(stat_filename, 'w+', 0)
        start_time = time.time()

        best_val_acc = 0

        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time_epoch = time.time()
            for batch in self.iterate_minibatches(X_train, y_train, batch_size_train, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_val, y_val, batch_size_valid, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time_epoch))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

            if (val_acc / val_batches * 100 > best_val_acc):
                best_val_acc = val_acc / val_batches * 100

            stat_file.write("{}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\n".format(
                epoch, time.time() - start_time, train_err / train_batches,
                val_err / val_batches, val_acc / val_batches * 100))

            if (time.time() - start_time > time_limit):
                break

        stat_file.close()

        return [val_err/val_batches, best_val_acc, time.time()-start_time, nparameters]

        # Optionally, you could now dump the network weights to a file like this:
        # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
        #
        # And load them again later on like this:
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)

    def write_new_line(self,filename,strng):
        f = open(filename,'w')
        f.write(strng)
        f.close()

    def append_new_line(self,filename,strng):
        f = open(filename,'a+')
        f.write(strng)
        f.close()

    def read_first_line(self,filename):
        f = open(filename,'r')
        line = f.readline()
        f.close()
        return line

    def read_last_line(self,filename):
        f = open(filename,'r')
        lines = f.readlines()
        if lines != []:
            lastline = lines[-1]
        else:
            lastline = ""
        f.close()
        return lastline

    def stopping_criterium(self, filename):
        return self.read_last_line(filename).strip() == "stop"

    def next_criterium(self, filename):
        return self.read_last_line(filename).strip() == "next"

    def main(self, worker_nb=0):
        ntrain = 50000          # the whole training set
        nvalid = 10000         #
        ntest = 10000           #
        batch_size_valid = 500  # does not influence training process, but reduces time loss from validation
        batch_size_test = 500   # same here
        #num_epochs = 100000     # to disable this stopping criterion
        #time_limit = 100000         # training time is limited to 60 seconds

        algorithm_type = 2  # SGD with momentum
        irun = 1  # one run only
        mexevaluations = 200
        nvariables = 4
        filename = "worker_{}".format(worker_nb)
        while(True):
            # Stopping criterium means that work is done
            if not(self.stopping_criterium(filename)):
                # If next criterium is satisfied, this worker should wait for the
                #   master to give a new configuration
                if not(self.next_criterium(filename)):
                    # Read the configuration line
                    line = self.read_first_line(filename)
                    if line.strip() == "wait":
                        pass
                    elif len(line.split()) == 6:
                        lst = line.split()
                        # Initialize values
                        nfilters = int(lst[0])
                        batch_size_train = int(lst[1])
                        M = float(lst[2])
                        LR = float(lst[3])
                        num_epochs = int(lst[4])
                        time_limit = float(lst[5])

                        print("nfilters: {}\tbatch_size_train: {}\t M: {:.6f}\t LR: {:.6f}".format(nfilters, batch_size_train, M, LR))

                        # Get result from ccn with given configuration
                        results = self.get_result(ntrain, nvalid, ntest, algorithm_type, batch_size_train, batch_size_valid, batch_size_test, num_epochs, "stat.txt", LR, M, nfilters, time_limit)

                        val_loss = results[0]
                        best_val_acc = results[1]
                        total_time = results[2]
                        nparameters = float(results[3])
                        # Check stopping criterium again
                        if not(self.stopping_criterium(filename)):
                            # Write configuration and result
                            self.write_new_line(filename,"{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(nfilters,batch_size_train,
                                    M,LR,num_epochs,time_limit,val_loss,best_val_acc,total_time,nparameters))
                            # Add "next" to last line so master knows it can give a new configuration
                            self.append_new_line(filename,"next")
                        else:
                            break
            else:
                break

# Initialize worker in command line with command: python worker.py 0, to initialize worker 0
if __name__ == '__main__':
    w = Worker()
    w.main(sys.argv[1])
