# -*- coding: utf-8 -*-

import sys
import numpy
from utils import *


class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label

        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)  # initialize bias 0


    def train(self, lr=0.1, input=None, L2_reg=0.00):        
        if input is not None:
            self.x = input

        p_y_given_x = self.output(self.x)
        d_y = self.y - p_y_given_x

        self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * numpy.mean(d_y, axis=0)
        self.d_y = d_y
        

    # def train(self, lr=0.1, input=None, L2_reg=0.00):
    #     self.forward(input)
    #     self.backward(lr, L2_reg)

    # def forward(self, input=None):
    #     if input is not None:
    #         self.x = input

    #     p_y_given_x = self.output(self.x)
    #     self.d_y = self.y - p_y_given_x
        
    # def backward(self, lr=0.1, L2_reg=0.00):
    #     self.W += lr * numpy.dot(self.x.T, self.d_y) - lr * L2_reg * self.W
    #     self.b += lr * numpy.mean(self.d_y, axis=0)


    def output(self, x):
        # return sigmoid(numpy.dot(x, self.W) + self.b)
        return softmax(numpy.dot(x, self.W) + self.b)

    def predict(self, x):
        return self.output(x)


    def negative_log_likelihood(self):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)
        sigmoid_activation = softmax(numpy.dot(self.x, self.W) + self.b)

        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
            (1 - self.y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


def test_lr(learning_rate=0.1, n_epochs=500):

    rng = numpy.random.RandomState(123)

    # training data
    d = 2
    N = 10
    x1 = rng.randn(N, d) + numpy.array([0, 0])
    x2 = rng.randn(N, d) + numpy.array([20, 10])
    y1 = [[1, 0] for i in xrange(N)]
    y2 = [[0, 1] for i in xrange(N)]

    x = numpy.r_[x1.astype(int), x2.astype(int)]
    y = numpy.r_[y1, y2]


    # construct LogisticRegression
    classifier = LogisticRegression(input=x, label=y, n_in=d, n_out=2)

    # train
    for epoch in xrange(n_epochs):
        classifier.train(lr=learning_rate)
        # cost = classifier.negative_log_likelihood()
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        learning_rate *= 0.995


    # test
    result = classifier.predict(x)
    for i in xrange(N):
        print result[i]
    print
    for i in xrange(N):
        print result[N+i]



if __name__ == "__main__":
    test_lr()
