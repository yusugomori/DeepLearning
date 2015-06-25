# -*- coding: utf-8 -*-

import sys
import numpy
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from utils import *


class MLP(object):
    def __init__(self, input, label, n_in, n_hidden, n_out, rng=None):

        self.x = input
        self.y = label

        if rng is None:
            rng = numpy.random.RandomState(1234)

        # construct hidden_layer (tanh or sigmoid so far)
        self.hidden_layer = HiddenLayer(input=self.x,
                                        n_in=n_in,
                                        n_out=n_hidden,
                                        rng=rng,
                                        activation=numpy.tanh)

        # construct log_layer (softmax)
        self.log_layer = LogisticRegression(input=self.hidden_layer.output,
                                            label=self.y,
                                            n_in=n_hidden,
                                            n_out=n_out)

    def train(self):
        layer_input = self.hidden_layer.forward()
        self.log_layer.train(input=layer_input)
        self.hidden_layer.backward(prev_layer=self.log_layer)
        

    def predict(self, x):
        x = self.hidden_layer.output(x)
        return self.log_layer.predict(x)


def test_mlp(n_epochs=100):

    x = numpy.array([[1,1,1,0,0,0],
                     [1,0,1,0,0,0],
                     [1,1,1,0,0,0],
                     [0,0,1,1,1,0],
                     [0,0,1,1,0,0],
                     [0,0,1,1,1,0]])
    y = numpy.array([[1, 0],
                     [1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [0, 1]])


    rng = numpy.random.RandomState(123)


    # construct MLP
    classifier = MLP(input=x, label=y, n_in=6, n_hidden=15, n_out=2, rng=rng)

    # train
    for epoch in xrange(n_epochs):
        classifier.train()


    # test
    x = numpy.array([[1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0],
                     [1, 1, 1, 1, 1, 0]])

    print classifier.predict(x)
        

if __name__ == "__main__":
    test_mlp()
