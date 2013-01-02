#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Logistic Regression
'''

import sys
import numpy
from utils import *


class LogisticRegression(object):
    def __init__(self, input=None, n_in=2, n_out=2):
        self.input = input
        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)          # initialize bias 0

    def train(self, y, lr=0.1):
        y_pred = softmax(numpy.dot(self.input, self.W) + self.b)
        d_y = y - y_pred

        self.W += lr * numpy.dot(self.input.T, d_y)
        self.b += lr * numpy.mean(d_y, axis=0)

        # cost = self.negative_log_likelihood()
        # return cost

    def negative_log_likelihood(self):
        sigmoid_activation = sigmoid(numpy.dot(self.input, self.W) + self.b)
        
        # entropy = - numpy.mean(numpy.sum(self.input * numpy.log(sigmoid_activation), axis=1))
        # return entropy
        
        cross_entropy = - numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation) +
            (1 - self.input) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


    def predict(self, x):
        return sigmoid(numpy.dot(x, self.W) + self.b)


def test_lr(learning_rate=0.1, n_epochs=100):
    n_epochs = 10


    # training data
    rng = numpy.random.RandomState(123)
    n_each = 10
    m1 = -5.
    s1 = 1.0
    m2 = 10.
    s2 = 10.

    x = []
    y = []
    
    for i in xrange(n_each):
        x.append([rng.normal(m1, s1), rng.normal(m1, s1)])
        y.append([0])

    for i in xrange(n_each):
        x.append([rng.normal(m2, s2), rng.normal(m2, s2)])
        y.append([1])
    
    x = numpy.array(x)
    y = numpy.array(y)



    # construct LogisticRegression
    classifier = LogisticRegression(input=x, n_in=2, n_out=1)

    # train
    for epoch in xrange(n_epochs):
        classifier.train(y=y, lr=learning_rate)
        # cost = classifier.negative_log_likelihood()
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost


    # test
    x = numpy.array([-5.0, 1])
    print >> sys.stderr, classifier.predict(x)


if __name__ == "__main__":
    test_lr()
