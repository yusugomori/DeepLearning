''' '''
import numpy

numpy.seterr(all='ignore')


def sigmoid(x):
    # numpy.seterrcall(sigmoid_err_handler)  # overflow handling
    # numpy.seterr(all='call')
    return 1. / (1 + numpy.exp(-x))


def sigmoid_err_handler(type, flg):
    # Log
    return


def softmax(x):
    e = numpy.exp(x - numpy.max(x, axis=0))  # prevent overflow
    return e / numpy.sum(e)
