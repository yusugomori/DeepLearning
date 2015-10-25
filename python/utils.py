
import numpy
numpy.seterr(all='ignore')


def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


def dsigmoid(x):
    return x * (1. - x)

def tanh(x):
    return numpy.tanh(x)

def dtanh(x):
    return 1. - x * x

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2


def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)


# # probability density for the Gaussian dist
# def gaussian(x, mean=0.0, scale=1.0):
#     s = 2 * numpy.power(scale, 2)
#     e = numpy.exp( - numpy.power((x - mean), 2) / s )

#     return e / numpy.square(numpy.pi * s)

# for CNN
def create_demo_data(N_each, channel, n_in, n_out, rng, p=0.9):
    if rng is None:
        rng = numpy.random.RandomState(1234)

    data = numpy.zeros( (N_each * n_out, channel, n_in, n_in) )
    label = numpy.zeros( (N_each * n_out, n_out) )

    K = n_in / n_out

    index = 0
    for k in xrange(n_out):  # for each class        
        for num in xrange(N_each):  # for each sub data
            for c in xrange(channel):
                for i in xrange(n_in):
                    for j in xrange(n_in):                

                        if i < (k+1) * K and i >= k * K:
                            # a = int(128 * rng.rand() + 128) * rng.binomial(size=1, n=1, p=p) / 256.0
                            a = 128.0 * rng.binomial(size=1, n=1, p=p) / 256.0

                        else:
                            a = 128.0 * rng.binomial(size=1, n=1, p=1-p) / 256.0


                        data[index][c][i][j] = a

            for i in xrange(n_out):
                if i == k:
                    label[index][i] = 1.0
                else:
                    label[index][i] = 0.0

            index += 1

    return data, label


