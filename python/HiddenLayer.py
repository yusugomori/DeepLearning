# -*- coding: utf-8 -*-

import sys
import numpy
from utils import *


class HiddenLayer(object):
    def __init__(self, input, n_in, n_out,\
                 W=None, b=None, rng=None, activation=numpy.tanh):
        
        if rng is None:
            rng = numpy.random.RandomState(1234)

        if W is None:
            a = 1. / n_in
            W = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_in, n_out)))

        if b is None:
            b = numpy.zeros(n_out)  # initialize bias 0

        self.rng = rng
        self.x = input
        self.W = W
        self.b = b

        if activation == numpy.tanh:
            self.dactivation = dtanh
        elif activation == sigmoid:
            self.dactivation = dsigmoid
        else:
            raise ValueError('activation function not supported.')
        
        self.activation = activation
        


    def output(self, input=None):
        if input is not None:
            self.x = input
        
        linear_output = numpy.dot(self.x, self.W) + self.b

        return (linear_output if self.activation is None
                else self.activation(linear_output))


    def sample_h_given_v(self, input=None):
        if input is not None:
            self.x = input

        v_mean = self.output()
        h_sample = self.rng.binomial(size=v_mean.shape,
                                           n=1,
                                           p=v_mean)
        return h_sample



    def forward(self, input=None):
        return self.output(input=input)


    def backward(self, prev_layer, lr=0.1, input=None):
        if input is not None:
            self.x = input

        # d_y = (1 - prev_layer.x * prev_layer.x) * numpy.dot(prev_layer.d_y, prev_layer.W.T)
        d_y = self.dactivation(prev_layer.x) * numpy.dot(prev_layer.d_y, prev_layer.W.T)

        self.W += lr * numpy.dot(self.x.T, d_y)
        self.b += lr * numpy.mean(d_y, axis=0)

        self.d_y = d_y
