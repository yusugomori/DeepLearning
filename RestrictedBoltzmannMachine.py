#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 Restricted Boltzmann Machine (RBM)

 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007


   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials


"""

import sys
import numpy
from utils import *

class RestrictedBoltzmannMachine(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, \
        W=None, hbias=None, vbias=None, numpy_rng=None):
        
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)


        if W is None:
            a = 1. / n_visible
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.zeros(n_visible)  # initialize v bias 0


        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias

        self.params = [self.W, self.hbias, self.vbias]


    def contrastive_divergence(self, lr=0.1, k=1):
        ''' CD-k '''
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        chain_start = ph_sample

        for step in xrange(k):
            if step == 0:
                pre_sigmoid_nvs, nv_means, nv_samples,\
                pre_sigmoid_nhs, nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                pre_sigmoid_nvs, nv_means, nv_samples,\
                pre_sigmoid_nhs, nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        # chain_end = nv_samples

        self.W += lr * (numpy.dot(self.input, ph_sample) - numpy.dot(nv_samples, nh_samples))
        self.hbias += lr * numpy.mean(ph_sample - nh_samples, axis=0)
        self.vbias += lr * numpy.mean(self.input - nv_samples, axis=1)


        cost = self.get_reconstruction_cross_entropy()
        return cost


    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.numpy_rng.binomial(size=h1_mean.shape,   # discrete: binomial
                                       n=1,
                                       p=h1_mean)

        return [pre_sigmoid_h1, h1_mean, h1_sample]


    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.numpy_rng.binomial(size=v1_mean.shape,   # discrete: binomial
                                            n=1,
                                            p=v1_mean)
        
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def propup(self, v):
        pre_sigmoid_activation = numpy.dot(v, self.W) + self.hbias
        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]

    def propdown(self, h):
        pre_sigmoid_activation = numpy.dot(h, self.W.T) + self.vbias
        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]


    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]
    

    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation = numpy.dot(self.input, self.W) + self.hbias
        sigmoid_activation = sigmoid(pre_sigmoid_activation)

        pre_sigmoid_activation = - numpy.dot(sigmoid_activation, self.W.T) - self.vbias
        sigmoid_activation = sigmoid(pre_sigmoid_activation)

        cross_entropy =  numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation) +
            (1 - self.input) * numpy.log(1 - sigmoid_activation),
                      axis=1))
        
        return cross_entropy


    def free_energy(self, v_sample):
        wx_b = numpy.dot(v_sample, self.W) + self.hbias
        vbias_term = numpy.dot(v_sample, self.vbias)
        hidden_term = numpy.sum(numpy.log(1 + numpy.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term






def test_rbm(learning_rate=0.1, k=1, training_epochs=15):
    data = numpy.array([
        [1,1,1,0,0,0],
        [1,0,1,0,0,0],
        [1,1,1,0,0,0],
        [0,0,1,1,1,0],
        [0,0,1,1,0,0],
        [0,0,1,1,1,0]])
    # A 6x6 matrix where each row is a training example and each column is a visible unit.

    rng = numpy.random.RandomState(123)

    # construct RBM
    rbm = RestrictedBoltzmannMachine(input=data, n_visible=6, n_hidden=2, numpy_rng=rng)

    for epoch in xrange(training_epochs):
        cost = rbm.contrastive_divergence(lr=learning_rate, k=k)
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost



if __name__ == "__main__":
    test_rbm()
