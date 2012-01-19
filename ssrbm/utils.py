"""
"""
import numpy
from PIL import Image

import theano
from theano import tensor
floatX = theano.config.floatX

import theano_linear


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



def scale_to_unit_interval(ndar,eps=1e-8):
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / max(ndar.max(),eps)
    return ndar

def tile_raster_images(X, img_shape,
        tile_shape=None, tile_spacing=(1,1),
        scale_rows_to_unit_interval=True,
        output_pixel_vals=True,
        min_dynamic_range=1e-4,
        ):
    """
    Transform an array with one flattened image per row, into an array in which images are
    reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images, and also columns of
    matrices for transforming those rows (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can be 2-D ndarrays or None
    :param X: a 2-D array in which every row is a flattened image.
    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image
    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols) (Defaults to a square-ish
        shape with the right area for the number of images)
    :type min_dynamic_range: positive float
    :param min_dynamic_range: the dynamic range of each image is used in scaling to the unit
        interval, but images with less dynamic range than this will be scaled as if this were
        the dynamic range.

    :returns: array suitable for viewing as an image.  (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """
    # This is premature when tile_slices_to_image is not documented at all yet,
    # but ultimately true:
    #print >> sys.stderr, "WARN: tile_raster_images sucks, use tile_slices_to_image"
    if len(img_shape)==3 and img_shape[2]==3:
        # make this save an rgb image
        if scale_rows_to_unit_interval:
            print >> sys.stderr, "WARN: tile_raster_images' scaling routine messes up colour - try tile_slices_to_image"
        return tile_raster_images(
                (X[:,0::3], X[:,1::3], X[:,2::3], None),
                img_shape=img_shape[:2],
                tile_shape=tile_shape,
                tile_spacing=tile_spacing,
                scale_rows_to_unit_interval=scale_rows_to_unit_interval,
                output_pixel_vals=output_pixel_vals,
                min_dynamic_range=min_dynamic_range)

    if isinstance(X, tuple):
        n_images_in_x = X[0].shape[0]
    else:
        n_images_in_x = X.shape[0]

    if tile_shape is None:
        tile_shape = most_square_shape(n_images_in_x)

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    #out_shape is the shape in pixels of the returned image array
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp 
        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        if scale_rows_to_unit_interval:
            raise NotImplementedError()
        assert len(X) == 4
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0,0,0,255]
        else:
            channel_defaults = [0.,0.,0.,1.]

        for i in xrange(4):
            if X[i] is None:
                out_array[:,:,i] = numpy.zeros(out_shape,
                        dtype='uint8' if output_pixel_vals else out_array.dtype
                        )+channel_defaults[i]
            else:
                out_array[:,:,i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        H, W = img_shape
        Hs, Ws = tile_spacing

        out_scaling = 1
        if output_pixel_vals and str(X.dtype).startswith('float'):
            out_scaling = 255

        out_array = numpy.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)
        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        try:
                            this_img = scale_to_unit_interval(
                                    X[tile_row * tile_shape[1] + tile_col].reshape(img_shape),
                                    eps=min_dynamic_range)
                        except ValueError:
                            raise ValueError('Failed to reshape array of shape %s to shape %s'
                                    % (
                                        X[tile_row*tile_shape[1] + tile_col].shape
                                        , img_shape
                                        ))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    out_array[
                        tile_row * (H+Hs):tile_row*(H+Hs)+H,
                        tile_col * (W+Ws):tile_col*(W+Ws)+W
                        ] \
                        = this_img * out_scaling
        return out_array


def most_square_shape(N):
    """rectangle (height, width) with area N that is closest to sqaure
    """
    for i in xrange(int(numpy.sqrt(N)),0, -1):
        if 0 == N % i:
            return (i, N/i)

def save_tiled_raster_images(tiled_img, filename):
    """Save a a return value from `tile_raster_images` to `filename`.

    Returns the PIL image that was saved
    """
    if tiled_img.ndim==2:
        img = Image.fromarray( tiled_img, 'L')
    elif tiled_img.ndim==3:
        img = Image.fromarray(tiled_img, 'RGBA')
    else:
        raise TypeError('bad ndim', tiled_img)

    img.save(filename)
    return img

def tile_slices_to_image_uint8(X, tile_shape=None):
    if str(X.dtype) != 'uint8':
        raise TypeError(X)
    if tile_shape is None:
        #how many tile rows and cols
        (TR, TC) = most_square_shape(X.shape[0])
    H, W = X.shape[1], X.shape[2]

    Hs = H+1 #spacing between tiles
    Ws = W+1 #spacing between tiles

    trows, tcols= most_square_shape(X.shape[0])
    outrows = trows * Hs - 1
    outcols = tcols * Ws - 1
    out = numpy.zeros((outrows, outcols,3), dtype='uint8')
    tr_stride= 1+X.shape[1]
    for tr in range(trows):
        for tc in range(tcols):
            Xrc = X[tr*tcols+tc]
            if Xrc.ndim==2: # if no color channel make it broadcast
                Xrc=Xrc[:,:,None]
            #print Xrc.shape
            #print out[tr*Hs:tr*Hs+H,tc*Ws:tc*Ws+W].shape
            out[tr*Hs:tr*Hs+H,tc*Ws:tc*Ws+W] = Xrc
    img = Image.fromarray(out, 'RGB')
    return img

def tile_slices_to_image(X,
        tile_shape=None,
        scale_each=True,
        min_dynamic_range=1e-4):
    #always returns an RGB image
    def scale_0_255(x):
        xmin = x.min()
        xmax = x.max()
        return numpy.asarray(
                255 * (x - xmin) / max(xmax - xmin, min_dynamic_range),
                dtype='uint8')

    if scale_each:
        uintX = numpy.empty(X.shape, dtype='uint8')
        for i, Xi in enumerate(X):
            uintX[i] = scale_0_255(Xi)
        X = uintX
    else:
        X = scale_0_255(X)
    return tile_slices_to_image_uint8(X, tile_shape=tile_shape)
