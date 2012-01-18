"""
"""
import numpy

import theano
from theano import tensor
floatX = theano.config.floatX

# XXX: get from skdata
import pylearn.dataset_ops.cifar10
import pylearn.dataset_ops.tinyimages

import tlinear


def clip_ramp(x, left, right, dtype=None):
    """
    """
    x0, y0 = left
    x1, y1 = right
    if dtype is not None:
        x, x0, x1, y0, y1 = [tensor.cast(o, dtype)
                for o in [x, x0, x1, y0, y1]]
    relpos = (x - x0) / (x1 - x0)
    slope = (y1 - y0) / (x1 - x0)
    return tensor.switch( x < x0, y0,
            tensor.switch(x > x1, y1,
                y0 + relpos * slope))


def contrastive_cost(free_energy_fn, pos_v, neg_v):
    """
    :param free_energy_fn: lambda (TensorType matrix MxN) ->  TensorType vector of M free energies
    :param pos_v: TensorType matrix MxN of M "positive phase" particles
    :param neg_v: TensorType matrix MxN of M "negative phase" particles

    :returns: TensorType scalar that's the sum of the difference of free energies

    :math: \sum_i free_energy(pos_v[i]) - free_energy(neg_v[i])

    """
    return (free_energy_fn(pos_v) - free_energy_fn(neg_v)).sum()


def contrastive_grad(free_energy_fn, pos_v, neg_v, wrt, other_cost=0, consider_constant=[]):
    """
    :param free_energy_fn: lambda (TensorType matrix MxN) ->  TensorType vector of M free energies
    :param pos_v: positive-phase sample of visible units
    :param neg_v: negative-phase sample of visible units
    :param wrt: TensorType variables with respect to which we want gradients (similar to the
        'wrt' argument to tensor.grad)
    :param other_cost: TensorType scalar (should be the sum over a minibatch, not mean)

    :returns: TensorType variables for the gradient on each of the 'wrt' arguments


    :math: Cost = other_cost + \sum_i free_energy(pos_v[i]) - free_energy(neg_v[i])
    :math: d Cost / dW for W in `wrt`


    This function is similar to tensor.grad - it returns the gradient[s] on a cost with respect
    to one or more parameters.  The difference between tensor.grad and this function is that
    the negative phase term (`neg_v`) is considered constant, i.e. d `Cost` / d `neg_v` = 0.
    This is desirable because `neg_v` might be the result of a sampling expression involving
    some of the parameters, but the contrastive divergence algorithm does not call for
    backpropagating through the sampling procedure.

    Warning - if other_cost depends on pos_v or neg_v and you *do* want to backpropagate from
    the `other_cost` through those terms, then this function is inappropriate.  In that case,
    you should call tensor.grad separately for the other_cost and add the gradient expressions
    you get from ``contrastive_grad(..., other_cost=0)``

    """
    cost=contrastive_cost(free_energy_fn, pos_v, neg_v)
    if other_cost:
        cost = cost + other_cost
    return theano.tensor.grad(cost,
            wrt=wrt,
            consider_constant=consider_constant+[neg_v])


def l1(X):
    """
    :param X: TensorType variable

    :rtype: TensorType scalar

    :returns: the sum of absolute values of the terms in X

    :math: \sum_i |X_i|

    Where i is an appropriately dimensioned index.

    """
    return abs(X).sum()


def l2(X):
    """
    :param X: TensorType variable

    :rtype: TensorType scalar

    :returns: the sum of absolute values of the terms in X

    :math: \sqrt{ \sum_i X_i^2 }

    Where i is an appropriately dimensioned index.

    """
    return TT.sqrt((X**2).sum())


_ndarray_status_fmt='%(msg)s shape=%(shape)s min=%(min)f mean=%(mean)f max=%(max)f'

def ndarray_status(x, fmt=_ndarray_status_fmt, msg="", **kwargs):
    kwargs.update(dict(
            msg=msg,
            min=x.min(),
            max=x.max(),
            mean=x.mean(),
            var = x.var(),
            shape=x.shape))
    return fmt%kwargs


def safe_update(a, b):
    """Performs union of a and b without replacing elements of 'a'
    """
    for k,v in dict(b).iteritems():
        if k in a:
            raise KeyError(k)
        a[k] = v
    return a


def sgd_updates(params, grads, stepsizes):
    """Return a list of (pairs) that can be used as updates in theano.function to implement
    stochastic gradient descent.

    :param params: variables to adjust in order to minimize some cost
    :type params: a list of variables (theano.function will require shared variables)
    :param grads: the gradient on each param (with respect to some cost)
    :type grads: list of theano expressions
    :param stepsizes: step by this amount times the negative gradient on each iteration
    :type stepsizes: [symbolic] scalar or list of one [symbolic] scalar per param
    """
    try:
        iter(stepsizes)
    except TypeError:
        stepsizes = [stepsizes for p in params]
    if len(params) != len(grads):
        raise ValueError('params and grads have different lens')
    updates = [(p, p - step * gp) for (step, p, gp) in zip(stepsizes, params, grads)]
    return updates


def sgd_momentum_updates(params, grads, stepsizes, momentum=0.9):
    """
    XXX
    """
    # if stepsizes is just a scalar, expand it to match params
    try:
        iter(stepsizes)
    except TypeError:
        stepsizes = [stepsizes for p in params]
    try:
        iter(momentum)
    except Exception:
        momentum = [momentum for p in params]
    if len(params) != len(grads):
        raise ValueError('params and grads have different lens')
    headings = [theano.shared(p.get_value(borrow=False) * 0) for p in params]
    updates = []
    for s, p, gp, m, h in zip(stepsizes, params, grads, momentum, headings):
        updates.append((p, p + s * h))
        updates.append((h, m * h - (1 - m) * gp))
    return updates


def sharedX(X, name):
    return shared(numpy.asarray(X, dtype=floatX), name=name)


def triangle_window(N, channels=3):
    """Return a triangle filter (1,channels,N,N)
    """
    w = numpy.arange(N)+1
    w = numpy.minimum(w, w[::-1])
    w = (w * w[:,None] * numpy.ones(channels)[:,None,None])[None,:,:,:]
    w = w.astype(floatX) / w.sum()
    return w


def gaussian_window(N, channels=3):
    if N==9:
        w = numpy.exp(-0.1 * (numpy.arange(N)-4)**2)
    elif N==7:
        w = numpy.exp(-0.2 * (numpy.arange(N)-3)**2)
    elif N==5:
        w = numpy.exp(-0.4 * (numpy.arange(N)-2)**2)
    else:
        raise NotImplementedError()
    w = (w * w[:,None] * numpy.ones(channels)[:,None,None])[None,:,:,:]
    w = w.astype(floatX) / w.sum()
    return w


def local_contrast_normalize(X, window, img_shape):
    """Return normalized X and the convolution transform
    """
    batchsize, channels, R, C = img_shape
    assert window.shape[0] == 1
    assert window.shape[1] == channels
    N = window.shape[2]
    assert window.shape[3] == N
    blur = tlinear.Conv2d(
            filters=sharedX(window, 'LCN_window'),
            img_shape=img_shape,
            border_mode='full')
    N2 = N//2
    # remove global mean
    X = X - X.mean(axis=[1, 2, 3]).dimshuffle(0, 'x', 'x', 'x')

    #remove local mean
    blurred_x = tensor.addbroadcast(blur.lmul(X), 1)
    x2c = X - blurred_x[:, :, N2:R + N2, N2:C + N2]

    # standardize contrast
    blurred_x2c_sqr = tensor.addbroadcast(blur.lmul(x2c ** 2), 1)
    x2c_lcn =  x2c / tensor.sqrt((10 + blurred_x2c_sqr[:, :, N2:R + N2, N2:C + N2]))

    return x2c_lcn, blur


def get_symbolic_data(dataset, s_idx, batchsize, split, loop, preprocessing, LCN_width=None):
    batch_range = s_idx * batchsize + numpy.arange(batchsize)
    if dataset=='cifar10':
        batch_x, batch_y = pylearn.dataset_ops.cifar10.cifar10(
                batch_range,
                split=split,
                dtype='uint8',
                rasterized=False,
                color='rgb',
                loop=loop)
    elif dataset=='tinyimages':
        assert split=='train'
        batch_x = pylearn.dataset_ops.tinyimages.tinyimages_op(
                batch_range,
                loop=loop,
                shuffled=True)
        batch_y = None
    else:
        raise ValueError('dataset', dataset)

    batch_x = tensor.cast(
                batch_x.dimshuffle(0, 3, 1, 2), # conv2d-friendly shape: batch channel rows cols
                floatX)

    if preprocessing == 'RAW':
        visible_pos_phase = batch_x
    if preprocessing == 'GCN': # global contrast normalization
        x2 = tensor.cast(batch_x.flatten(2), 'float32')
        x2c = x2 - x2.mean(axis=1).dimshuffle(0,'x')
        x2lcn = x2c / tensor.sqrt((10 + (x2c**2).mean(axis=1).dimshuffle(0,'x')))
        visible_pos_phase = x2lcn.reshape((batchsize, 3, 32, 32)) #conv2d-friendly
    elif preprocessing == 'ZCA':
        raise NotImplementedError()
    elif preprocessing == 'LCN': #local contrast normalization
        visible_pos_phase, blur_T = local_contrast_normalize(batch_x,
                gaussian_window(LCN_width, channels=3),
                img_shape=(batchsize, 3, 32, 32))
    else:
        raise ValueError('preprocessing', preprocessing)
    return visible_pos_phase, batch_y, batch_x

