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
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2
