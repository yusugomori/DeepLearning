# -*- coding: utf-8 -*-

import sys
import numpy
from utils import *

class ConvPoolLayer(object):
    def __init__(self, 
                 N,
                 image_size,
                 channel,   
                 n_kernel,  
                 kernel_size,
                 pool_size=[2, 2],
                 rng=None, activation=ReLU):

        if rng is None:
            rng = numpy.random.RandomState(1234)


        f_in = channel * kernel_size[0] * kernel_size[1]
        f_out = n_kernel * kernel_size[0] * kernel_size[1] / (pool_size[0] * pool_size[1])

        a = numpy.sqrt(6. / (f_in + f_out))

        W = numpy.array(rng.uniform(
            low=-a,
            high=a,
            size=(n_kernel, channel, kernel_size[0], kernel_size[1])  # filter_shape
            ))

        b = numpy.zeros(n_kernel)

        self.rng = rng

        self.image_size = image_size
        self.channel = channel
        self.n_kernel = n_kernel
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        self.W = W
        self.b = b

        if activation == tanh:
            self.dactivation = dtanh

        elif activation == sigmoid:
            self.dactivation = dsigmoid

        elif activation == ReLU:
            self.dactivation = dReLU

        else:
            raise ValueError('activation function not supported.')

        
        self.activation = activation


    def convolve(self, input):

        minibatch_size = len(input)

        s0 = self.image_size[0] - self.kernel_size[0] + 1
        s1 = self.image_size[1] - self.kernel_size[1] + 1

        convolved_input = numpy.zeros( (minibatch_size, self.n_kernel, s0, s1) )
        activated_input = numpy.zeros( (minibatch_size, self.n_kernel, s0, s1) )


        sum0 = 0.0
        sum1 = 0.0

        for batch in xrange(minibatch_size):

            for k in xrange(self.n_kernel):
                for i in xrange(s0):
                    for j in xrange(s1):

                        for c in xrange(self.channel):
                            for s in xrange(self.kernel_size[0]):
                                for t in xrange(self.kernel_size[1]):

                                    convolved_input[batch][k][i][j] += self.W[k][c][s][t] * input[batch][c][i+s][j+t]

                        activated_input[batch][k][i][j] = self.activation( convolved_input[batch][k][i][j] + self.b[k] )



        self.input = input
        self.convolved_input = convolved_input
        self.activated_input = activated_input

        return activated_input


    def dconvolve(self, prev_layer_delta, layer_input, learning_rate):

        minibatch_size = len(prev_layer_delta)
        
        s0 = self.image_size[0] - self.kernel_size[0] + 1
        s1 = self.image_size[1] - self.kernel_size[1] + 1

        delta = numpy.zeros( (minibatch_size, self.channel, self.image_size[0], self.image_size[1] ) )
        
        grad_W = numpy.zeros( (self.n_kernel, self.channel, self.kernel_size[0], self.kernel_size[1]) )
        grad_b = numpy.zeros( self.n_kernel )

        
        # calc gradients
        for batch in xrange(minibatch_size):
            for k in xrange(self.n_kernel):

                for i in xrange(s0):
                    for j in xrange(s1):

                        d = prev_layer_delta[batch][k][i][j] * self.dactivation(self.convolved_input[batch][k][i][j] + self.b[k])

                        grad_b[k] += d

                                    
                        for c in xrange(self.channel):
                            for s in xrange(self.kernel_size[0]):
                                for t in xrange(self.kernel_size[1]):

                                    grad_W[k][c][s][t] += d * self.input[batch][c][i+s][j+t]

        # udpate params
        for k in xrange(self.n_kernel):

            self.b[k] -= learning_rate * grad_b[k] / minibatch_size
            
            for c in xrange(self.channel):
                for s in xrange(self.kernel_size[0]):
                    for t in xrange(self.kernel_size[1]):
                        self.W[k][c][s][t] -= learning_rate * grad_W[k][c][s][t] / minibatch_size


        # calc delta
        for batch in xrange(minibatch_size):
            for c in xrange(self.channel):
                for i in xrange(self.image_size[0]):
                    for j in xrange(self.image_size[1]):

                        for k in xrange(self.n_kernel):
                            for s in xrange(self.kernel_size[0]):
                                for t in xrange(self.kernel_size[1]):
                                    
                                    if (i - (self.kernel_size[0] - 1) - s < 0) or (j - (self.kernel_size[1] - 1) - t < 0):
                                        d = 0
                                    else:
                                        d = prev_layer_delta[batch][k][i-(self.kernel_size[0]-1)-s][j-(self.kernel_size[1]-1)-t] * self.dactivation(self.convolved_input[batch][k][i-(self.kernel_size[0]-1)-s][j-(self.kernel_size[1]-1)-t] + self.b[k]) * self.W[k][c][s][t]

                                    delta[batch][c][i][j] += d


        return delta



    def maxpooling(self, input):

        minibatch_size = len(input)
        
        s0 = len(input[0][0][0]) / self.pool_size[0]
        s1 = len(input[0][0][1]) / self.pool_size[1]


        pooled_input = numpy.zeros( (len(input), self.n_kernel, s0, s1) )


        for batch in xrange(minibatch_size):
            for k in xrange(self.n_kernel):

                for i in xrange(s0):
                    for j in xrange(s1):

                        for s in xrange(self.pool_size[0]):
                            for t in xrange(self.pool_size[1]):

                                if s == 0 and t == 0:
                                    max_ = input[batch][k][self.pool_size[0]*i][self.pool_size[1]*j]
                                    next

                                if max_ < input[batch][k][self.pool_size[0]*i+s][self.pool_size[1]*j+t]:
                                    max_ = input[batch][k][self.pool_size[0]*i+s][self.pool_size[1]*j+t]


                        pooled_input[batch][k][i][j] = max_

        self.pooled_input = pooled_input
        return pooled_input




    def dmaxpooling(self, prev_layer_delta, layer_input, delta_size):

        minibatch_size = len(prev_layer_delta)
        
        s0 = len(prev_layer_delta[0][0])
        s1 = len(prev_layer_delta[0][0][0])

        delta = numpy.zeros( (minibatch_size, self.n_kernel, delta_size[0], delta_size[1]) )

        for batch in xrange(minibatch_size):
            for k in xrange(self.n_kernel):

                for i in xrange(s0):
                    for j in xrange(s1):

                        for s in xrange(self.pool_size[0]):
                            for t in xrange(self.pool_size[1]):
                                
                                if self.pooled_input[batch][k][i][j] == layer_input[batch][k][self.pool_size[0]*i+s][self.pool_size[1]*j+t]:
                                    d = prev_layer_delta[batch][k][i][j]
                                    
                                else:
                                    d = 0.0

                                delta[batch][k][self.pool_size[0]*i+s][self.pool_size[1]*j+t] = d


        return delta

        

    def output(self, input=None):
        
        convolved_X = self.convolve(input)
        pooled_X = self.maxpooling(convolved_X)

        return pooled_X
        

    def forward(self, input=None):
        return self.output(input)


    def backward(self, prev_layer_delta, conv_size, learning_rate):
                
        delta_pool = self.dmaxpooling(prev_layer_delta, self.activated_input, [ conv_size[0]*self.pool_size[0], conv_size[1]*self.pool_size[1] ] )

        delta_conv = self.dconvolve(delta_pool, self.input, learning_rate)

        return delta_conv
