# -*- coding: utf-8 -*-

import sys
import numpy
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from ConvPoolLayer import ConvPoolLayer
# from MLP import MLP
from utils import *


class CNN(object):
    def __init__(self, N, label, n_hidden, n_out, image_size, channel, n_kernels, kernel_sizes, pool_sizes, rng=None, activation=ReLU):

        if rng is None:
            rng = numpy.random.RandomState(1234)

        self.N = N
        self.n_hidden = n_hidden
        
        self.n_kernels = n_kernels

        self.pool_sizes = pool_sizes

        self.conv_layers = []
        self.conv_sizes = []


        # construct 1st conv_layer
        conv_layer0 = ConvPoolLayer(N, image_size, channel, n_kernels[0], kernel_sizes[0], pool_sizes[0], rng, activation)
        self.conv_layers.append(conv_layer0)

        conv_size = [ (image_size[0] - kernel_sizes[0][0] + 1) / pool_sizes[0][0], (image_size[1] - kernel_sizes[0][1] + 1) / pool_sizes[0][1] ]        
        self.conv_sizes.append(conv_size)


        # construct 2nd conv_layer
        conv_layer1 = ConvPoolLayer(N, conv_size, n_kernels[0], n_kernels[1], kernel_sizes[1], pool_sizes[1], rng, activation)
        self.conv_layers.append(conv_layer1)

        conv_size = [ (conv_size[0] - kernel_sizes[1][0] + 1) / pool_sizes[1][0], (conv_size[1] - kernel_sizes[1][0] + 1) / pool_sizes[1][1] ]
        self.conv_sizes.append(conv_size)

        # construct hidden_layer
        self.hidden_layer = HiddenLayer(None, n_kernels[-1] * conv_size[0] * conv_size[1], n_hidden, None, None, rng, activation)

        # construct log_layer
        self.log_layer = LogisticRegression(None, label, n_hidden, n_out)

        
    # def train(self, epochs, learning_rate, input=None):
    def train(self, epochs, learning_rate, input, test_input=None):

        for epoch in xrange(epochs):

            if (epoch + 1) % 5 == 0:
                print 'iter = %d/%d' %(epoch+1, epochs)

                print
                print '------------------'
                print 'TEST PROCESSING...'

                print self.predict(test_input)
                print '------------------'
                print
            
            # forward first conv layer 
            pooled_X = self.conv_layers[0].forward(input=input)

            # forward second conv layer 
            pooled_X = self.conv_layers[1].forward(input=pooled_X)

            # flatten input
            layer_input = self.flatten(pooled_X)

            # forward hidden layer
            layer_input = self.hidden_layer.forward(input=layer_input)

            # forward & backward logistic layer
            self.log_layer.train(lr=learning_rate, input=layer_input)

            # backward hidden layer
            self.hidden_layer.backward(prev_layer=self.log_layer, lr=learning_rate)


            flatten_size = self.n_kernels[-1] * self.conv_sizes[-1][0] * self.conv_sizes[-1][1]
            delta_flatten = numpy.zeros( (self.N, flatten_size) )

            for n in xrange(self.N):
                for i in xrange(flatten_size):

                    for j in xrange(self.n_hidden):
                        delta_flatten[n][i] += self.hidden_layer.W[i][j] * self.hidden_layer.d_y[n][j]

            # unflatten delta
            delta = numpy.zeros( (len(delta_flatten), self.n_kernels[-1], self.conv_sizes[-1][0], self.conv_sizes[-1][1]) )

            for n in xrange(len(delta)):
                index = 0
                for k in xrange(self.n_kernels[-1]):
                    for i in xrange(self.conv_sizes[-1][0]):
                        for j in xrange(self.conv_sizes[-1][1]):
                            delta[n][k][i][j] = delta_flatten[n][index]
                            index += 1

            # backward second conv layer
            delta = self.conv_layers[1].backward(delta, self.conv_sizes[1], learning_rate)

            # backward first conv layer
            self.conv_layers[0].backward(delta, self.conv_sizes[0], learning_rate)
            


    def flatten(self, input):
        
        flatten_size = self.n_kernels[-1] * self.conv_sizes[-1][0] * self.conv_sizes[-1][1]
        flattened_input = numpy.zeros((len(input), flatten_size))

        for n in xrange(len(flattened_input)):
            index = 0

            for k in xrange(self.n_kernels[-1]):
                for i in xrange(self.conv_sizes[-1][0]):
                    for j in xrange(self.conv_sizes[-1][1]):
                        flattened_input[n][index] = input[n][k][i][j]
                        index += 1

        # print flattened_input

        return flattened_input


    def predict(self, x):

        pooled_X = self.conv_layers[0].forward(input=x)

        pooled_X = self.conv_layers[1].forward(input=pooled_X)

        layer_input = self.flatten(pooled_X)

        x = self.hidden_layer.output(input=layer_input)

        return self.log_layer.predict(x)




def test_cnn():

    rng = numpy.random.RandomState(1234)

    K = 3
    N = 50
    M = 10
    train_N = N * K
    test_N = M * K
    image_size = [12, 12]
    channel = 1

    n_kernels = [10, 20]
    kernel_sizes = [[3, 3], [2, 2]]
    pool_sizes = [[2, 2], [2, 2]]
    n_hidden = 20

    n_out = K

    epochs = 500
    learning_rate = 0.1


    # create demo data
    X, Tx = create_demo_data( N, channel, image_size[0], K, rng, p=0.95)
    Y, Ty = create_demo_data( M, channel, image_size[0], K, rng, p=0.9)


    # construct CNN
    print 'Building the model...'
    classifier = CNN(train_N, Tx, n_hidden, n_out, image_size, channel, n_kernels, kernel_sizes, pool_sizes, rng, ReLU)
    

    # train
    print 'Training the model...'
    classifier.train(epochs, learning_rate, X, Y)


    # test
    print 'Testing the model'
    print classifier.predict(Y)




if __name__ == "__main__":
    test_cnn()
