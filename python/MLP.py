# -*- coding: utf-8 -*-

import sys
import numpy
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from utils import *


class MLP(object):
    def __init__(self, input, label, n_in, n_hidden, n_out, rng=None):
        """
        n_hidden: python list represent the hidden dimention 
        """

        self.x = input
        self.y = label

        if rng is None:
            rng = numpy.random.RandomState(1234)

        # construct hidden_layer
        layers_dim = numpy.hstack([n_in,n_hidden])
        self.hidden_layer = []

        for hidden_idx in xrange(len(layers_dim) - 1):
            self.hidden_layer.append(HiddenLayer(input=self.x, 
                                                 n_in=layers_dim[hidden_idx], 
                                                 n_out=layers_dim[hidden_idx+1], 
                                                 rng=rng,
                                                 activation=tanh))

        # construct log_layer
        self.log_layer = LogisticRegression(input=self.hidden_layer[-1].output,
                                            label=self.y,
                                            n_in=n_hidden[-1],
                                            n_out=n_out)


    def train(self):
        # forward hidden_layer
        layer_input = self.x

        for hidden_idx in range(len(self.hidden_layer)):

            layer_input = self.hidden_layer[hidden_idx].forward(input=layer_input)

        # forward & backward log_layer
        self.log_layer.train(input=layer_input)

        # backward hidden_layer
        for hidden_idx in range(len(self.hidden_layer))[::-1]:

            if hidden_idx == len(self.hidden_layer) - 1:

                self.hidden_layer[hidden_idx].backward(prev_layer=self.log_layer)

                continue

            self.hidden_layer[hidden_idx].backward(prev_layer=self.hidden_layer[hidden_idx+1])


    def predict(self, x):
        for hidden_idx in range(len(self.hidden_layer)):
            x = self.hidden_layer[hidden_idx].output(input=x)

        return self.log_layer.predict(x)


def test_mlp(n_epochs=5000):

    x = numpy.array([[0,  0],
                     [0,  1],
                     [1,  0],
                     [1,  1]])

    y = numpy.array([[0, 1],
                     [1, 0],
                     [1, 0],
                     [0, 1]])


    rng = numpy.random.RandomState(123)


    # construct MLP
    classifier = MLP(input=x, label=y, n_in=2, n_hidden=[3,4], n_out=2, rng=rng)

    # train
    for epoch in xrange(n_epochs):
        classifier.train()


    # test
    print classifier.predict(x)
        

if __name__ == "__main__":
    test_mlp()
