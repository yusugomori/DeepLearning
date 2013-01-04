#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO

'''
 Multi-Layer Perceptron

 References :
   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials

 
'''

import sys
import numpy
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from utils import *

class MLP(object):
    def __init__(self, input, n_in, n_hidden, n_out, numpy_rng=None):
        self.hidden_layer = HiddenLayer(input=input, n_in=n_in, n_out=n_hidden,\
                                        numpy_rng=numpy_rng, activation=numpy.tanh)

        ''' ここから下は need update  '''

        # *1 と連動  input が毎回変わる？
        #  LR内で self.input にすべきでない？
        #  or if input then self.input = input
        self.lr_layer = LogisticRegression(input=self.hidden_layer.output(),
                                           n_in=n_hidden,
                                           n_out=n_out)


        # L1 norm
        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.lr_layer.W).sum()
        # print self.L1

        # L2 norm
        self.L2_sqr = (self.hidden_layer.W ** 2).sum() + (self.lr_layer.W ** 2).sum()
        

        self.negative_log_likelihood = self.lr_layer.negative_log_likelihood
        

    def train(self, y, lr=0.1):
        # input to hidden
        # hidden to output
        
        print 'train'
        # numpy.dot(self.input, self.W) + self.b)        
        # y_pred = 
        


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100, n_hidden=3):
    # XOR
    x = numpy.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])
    y = numpy.array([[1],
                     [0],
                     [0],
                     [1]])

    rng = numpy.random.RandomState(1234)

    # construct MLP
    classifier = MLP(input=x, n_in=2, n_hidden=3, n_out=1, numpy_rng=rng)
    

if __name__ == "__main__":
    test_mlp()
