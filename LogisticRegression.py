#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Logistic Regression
 
 References :
   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials


'''

import sys
import numpy
from utils import *


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.input = input
        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)          # initialize bias 0

        # self.params = [self.W, self.b]

    def train(self, y, lr=0.1, input=None):
        if input is not None:
            self.input = input

        p_y_given_x = softmax(numpy.dot(self.input, self.W) + self.b)
        d_y = y - p_y_given_x
        
        self.W += lr * numpy.dot(self.input.T, d_y)
        self.b += lr * numpy.mean(d_y, axis=0)

        
        # cost = self.negative_log_likelihood()
        # return cost

    def negative_log_likelihood(self, y):
        sigmoid_activation = softmax(numpy.dot(self.input, self.W) + self.b)

        # entropy = - numpy.mean(numpy.sum(y * numpy.log(sigmoid_activation), axis=1))
        # return entropy
        
        cross_entropy = - numpy.mean(
            numpy.sum(y * numpy.log(sigmoid_activation) +
            (1 - y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


    def predict(self, x):
        return sigmoid(numpy.dot(x, self.W) + self.b)


def test_lr(learning_rate=0.01, n_epochs=5):
    # training data
    x = numpy.array([[-10., -5.],
                     [-5., -10.],
                     [5., 10.],
                     [10., 5.]])
    y = numpy.array([[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [1, 1, 1, 1],
                     [1, 1, 1, 1]])


    # construct LogisticRegression
    classifier = LogisticRegression(input=x, n_in=2, n_out=4)

    # train
    for epoch in xrange(n_epochs):
        classifier.train(y=y, lr=learning_rate)
        cost = classifier.negative_log_likelihood(y=y)
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        learning_rate *= 0.95


    # test
    print 
    print 'test'
    x = numpy.array([-10., -5.])
    print >> sys.stderr, classifier.predict(x)  # 0
    x = numpy.array([10., 5.])
    print >> sys.stderr, classifier.predict(x)  # 1

if __name__ == "__main__":
    test_lr()
