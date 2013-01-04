#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Hidden Layer


 References :
   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials


'''

import sys
import numpy
from utils import *


class HiddenLayer(object):
    def __init__(self, input, n_in, n_out,\
                 W=None, b=None, numpy_rng=None, activation=numpy.tanh):
        
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if W is None:
            a = 1. / n_in
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_in, n_out)))

            W = initial_W

        if b is None:
            b = numpy.zeros(n_out)  # initialize bias 0


        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.b = b

        self.activation = activation
        
        # linear_output = numpy.dot(input, self.W) + self.b
        # self.output = (linear_output if activation is None
        #                else activation(linear_output))

        # self.params = [self.W, self.b]

    def output(self, input=None):
        if input is not None:
            self.input = input
        
        linear_output = numpy.dot(self.input, self.W) + self.b
        # linear_output = numpy.dot(self.input, self.W) + self.b
        return (linear_output if self.activation is None
                else self.activation(linear_output))


