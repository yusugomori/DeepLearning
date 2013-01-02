#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Deep Belief Nets
'''

import numpy
import LogisticRegression
import RestrictedBoltzmannMachine

# HiddenLayer ??



class DBN(object):
    def __init__(self, n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2, \
                 numpy_rng=None):   # constructor does not contain input

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)


        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)
        

        assert self.n_layers > 0


        # construct multi layers
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]




def test_dbn():
    dbn = DBN()


if __name__ == "__main__":
    test_dbn()
