# -*- coding: utf-8 -*-

import sys
import numpy
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from utils import *


class Dropout(object):
    def __init__(self, input, label,\
                 n_in, hidden_layer_sizes, n_out,\
                 rng=None, activation=ReLU):

        self.x = input
        self.y = label

        self.hidden_layers = []
        self.n_layers = len(hidden_layer_sizes)
        
        if rng is None:
            rng = numpy.random.RandomState(1234)

        assert self.n_layers > 0


        # construct multi-layer 
        for i in xrange(self.n_layers):

            # layer_size
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layer_sizes[i-1]

            # layer_input
            if i == 0:
                layer_input = self.x

            else:
                layer_input = self.hidden_layers[-1].output()

            # construct hidden_layer
            hidden_layer = HiddenLayer(input=layer_input,
                                       n_in=input_size,
                                       n_out=hidden_layer_sizes[i],
                                       rng=rng,
                                       activation=activation)
            
            self.hidden_layers.append(hidden_layer)


        # layer for ouput using Logistic Regression (softmax)
        self.log_layer = LogisticRegression(input=self.hidden_layers[-1].output(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_out)


    def train(self, epochs=5000, dropout=True, p_dropout=0.5, rng=None):

        for epoch in xrange(epochs):
            dropout_masks = []  # create different masks in each training epoch

            # forward hidden_layers
            for i in xrange(self.n_layers):
                if i == 0:
                    layer_input = self.x

                layer_input = self.hidden_layers[i].forward(input=layer_input)

                if dropout == True:
                    mask = self.hidden_layers[i].dropout(input=layer_input, p=p_dropout, rng=rng)
                    layer_input *= mask

                    dropout_masks.append(mask)


            # forward & backward log_layer
            self.log_layer.train(input=layer_input)


            # backward hidden_layers
            for i in reversed(xrange(0, self.n_layers)):
                if i == self.n_layers-1:
                    prev_layer = self.log_layer
                else:
                    prev_layer = self.hidden_layers[i+1]

                if dropout == True:
                    self.hidden_layers[i].backward(prev_layer=prev_layer, dropout=True, mask=dropout_masks[i])
                else:
                    self.hidden_layers[i].backward(prev_layer=prev_layer)
                


    def predict(self, x, dropout=True, p_dropout=0.5):
        layer_input = x

        for i in xrange(self.n_layers):
            if dropout == True:
                self.hidden_layers[i].W = (1 - p_dropout) * self.hidden_layers[i].W
            
            layer_input = self.hidden_layers[i].output(input=layer_input)

        return self.log_layer.predict(layer_input)



def test_dropout(n_epochs=5000, dropout=True, p_dropout=0.5):

    x = numpy.array([[0,  0],
                     [0,  1],
                     [1,  0],
                     [1,  1]])

    y = numpy.array([[0, 1],
                     [1, 0],
                     [1, 0],
                     [0, 1]])

    rng = numpy.random.RandomState(123)


    # construct Dropout MLP
    classifier = Dropout(input=x, label=y, \
                         n_in=2, hidden_layer_sizes=[10, 10], n_out=2, \
                         rng=rng, activation=ReLU)


    # train XOR
    classifier.train(epochs=n_epochs, dropout=dropout, \
                     p_dropout=p_dropout, rng=rng)


    # test
    print classifier.predict(x)



if __name__ == "__main__":
    test_dropout()
