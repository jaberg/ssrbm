import logging
logger = logging.getLogger("Tssrbm")
logger.setLevel(logging.INFO)

import numpy
import theano
from theano import shared, tensor
import pylearn.io.image_tiling
from tlinear import dot, dot_outshape

try:
    # download this from https://bitbucket.org/jaberg/theano-curand
    from theano_curand import CURAND_RandomStreams as RandomStreams
    print 'INFO: using CURAND rng'
except ImportError:
    try:
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        print 'INFO: using MRG rng'
    except ImportError:
        print 'WARNING: falling back on numpy rng (SLOW)'
        RandomStreams = tensor.shared_randomstreams.RandomStreams

from utils import clip_ramp
from utils import contrastive_grad
from utils import l1
from utils import l2
from utils import ndarray_status
from utils import sgd_updates
from utils import sgd_momentum_updates
from utils import sharedX


floatX = theano.config.floatX


class RBM(object):
    def __init__(self, conf, numpy_rng, W, Lambda):
        """
        :param W: a LinearTransform instance for the weights.

        :param Lambda: a LinearTransform instance, parametrizing the h-dependent
        precision information regarding visibles.
        """
        self.conf = conf
        self.W = W
        self.Lambda = Lambda
        if Lambda:
            if W.col_shape() != Lambda.col_shape():
                raise ValueError('col_shape mismatch',
                        (W.col_shape(), Lambda.col_shape()))
            if W.row_shape() != Lambda.row_shape():
                raise ValueError('row_shape mismatch',
                        (W.row_shape(), Lambda.row_shape()))

        # Energy term has vW(sh), so...
        h_shp = self.h_shp = W.col_shape()
        s_shp = self.s_shp = W.col_shape()
        v_shp = self.v_shp = W.row_shape()
        logger.info("RBM Shapes h_shp=%s, s_shp=%s, v_shp=%s" %(h_shp, s_shp, v_shp))

        # alpha (precision on slab variables)
        alpha_init = numpy.zeros(s_shp)+conf['alpha0']
        if conf['alpha_irange']:
            alpha_init += (2 * numpy_rng.rand(*s_shp) - 1)*conf['alpha_irange']

        if conf['alpha_logdomain']:
            self.alpha = sharedX(numpy.log(alpha_init), name='alpha')
        else:
            self.alpha = sharedX(alpha_init, name='alpha')

        # mu (mean of slab vars)

        self.mu = sharedX(
                conf['mu0'] + numpy_rng.uniform(size=s_shp,
                    low=-conf['mu_irange'],
                    high=conf['mu_irange']),
                name='mu')

        # b (bias of spike vars)
        self.b = sharedX(
                conf['b0'] + numpy_rng.uniform(size=h_shp,
                    low=-conf['b_irange'],
                    high=conf['b_irange']),
                name='b')

        # B (precision on visible vars)
        if conf['B_full_diag']:
            B_init = numpy.zeros(v_shp) + conf['B0']
        else:
            B_init = numpy.zeros(()) + conf['B0']
        if conf['B_logdomain']:
            B_init = numpy.log(B_init)
        self.B = sharedX(B_init, name='B')

        self._params = [self.mu, self.B, self.b, self.alpha]

    def usable_alpha(self):
        if self.conf['alpha_logdomain']:
            return tensor.exp(self.alpha)
        else:
            return self.alpha

    def usable_beta(self):
        if self.conf['B_logdomain']:
            return tensor.exp(self.B)
        else:
            return self.B

    def mean_var_v_given_h_s(self, h,s):
        v_var = 1 / (self.usable_beta()
                + (ldot(h, self.Lambda.T) if self.Lambda else 0))
        v_mu = ldot(h*s, self.W.T) * v_var
        return v_mu, v_var

    def mean_var_s_given_v_h1(self, v):
        alpha = self.usable_alpha()
        return self.mu + ldot(v, self.W)/alpha, 1.0 / alpha

    def _mean_h_given_v(self, v):
        alpha = self.usable_alpha()
        return tensor.add(
                    self.b,
                    -0.5 * ldot(v * v, self.Lambda) if self.Lambda else 0,
                    self.mu * ldot(v, self.W),
                    0.5 * tensor.sqr(ldot(v, self.W))/alpha)

    def mean_h_given_v(self, v):
        egy = self._mean_h_given_v(v)
        return tensor.nnet.sigmoid(egy)

    def free_energy_given_v(self, v):
        sigmoid_arg = self._mean_h_given_v(v)
        hterm = tensor.sum(
                tensor.nnet.softplus(sigmoid_arg),
                axis=range(1,sigmoid_arg.ndim))
        return tensor.add(
                0.5 * tensor.sum(
                    self.usable_beta() * (v**2),
                    axis=range(1,v.ndim)),
                -hterm)

    def gibbs_step_for_v(self, v, rng, return_locals=False):
        #positive phase
        h_mean_shape = (self.conf['batchsize'],) + self.h_shp
        h_mean = self.mean_h_given_v(v)
        h_sample = tensor.cast(rng.uniform(size=h_mean_shape) < h_mean, floatX)
        s_mu, s_var = self.mean_var_s_given_v_h1(v)

        s_mu_shape = (self.conf['batchsize'],) + self.s_shp
        s_sample = s_mu + rng.normal(size=s_mu_shape) * tensor.sqrt(s_var)
        s_sample = s_sample * h_sample

        #negative phase
        vv_mean, vv_var = self.mean_var_v_given_h_s(h_sample, s_sample)
        vv_mean_shape = (self.conf['batchsize'],) + self.v_shp
        vv_sample = rng.normal(size=vv_mean_shape) * tensor.sqrt(vv_var) + vv_mean
        if return_locals:
            return vv_sample, locals()
        else:
            return vv_sample

    def cd_updates(self, pos_v, neg_v, lr, other_cost=0):
        grads = contrastive_grad(self.free_energy_given_v,
                pos_v, neg_v,
                wrt=self.params(),
                other_cost=other_cost)
        stepsizes=lr
        if self.conf.get('momentum', 0.0):
            logger.info('Using momentum %s'%self.conf['momentum'])
            rval = dict(
                    sgd_momentum_updates(
                        self.params(),
                        grads,
                        stepsizes=stepsizes,
                        momentum=self.conf['momentum']))
        else:
            rval = dict(
                    sgd_updates(
                        self.params(),
                        grads,
                        stepsizes=stepsizes))
        #DEBUG STORE GRADS
        grad_shared_vars = [sharedX(0*p.get_value(),'') for p in self.params()]
        self.grad_shared_vars = grad_shared_vars
        rval.update(dict(zip(grad_shared_vars, grads)))
        return rval

    def params(self):
        # return the list of *shared* learnable parameters
        # that are, in your judgement, typically learned in this model
        rval = list(self._params)
        rval += self.W.params()
        if self.Lambda:
            rval +=self.Lambda.params()
        return rval

    def print_status(self):
        print ndarray_status(self.b.get_value(), msg='b')
        print ndarray_status(self.mu.get_value(), msg='mu')
        if self.conf['alpha_logdomain']:
            print ndarray_status(numpy.exp(self.alpha.get_value()), msg='alpha')
        else:
            print ndarray_status(self.alpha.get_value(), msg='alpha')
        if self.conf['B_logdomain']:
            print ndarray_status(numpy.exp(self.B.get_value()), msg='B')
        else:
            print ndarray_status(self.B.get_value(), msg='B')


class Gibbs(object):
    def __init__(self, rbm, particles, rng):
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        seed=int(rng.randint(2**30))
        self.rbm = rbm
        self.n_particles = particles.shape[0]
        assert particles.shape[1:] == rbm.v_shp
        self.particles = sharedX(
            particles,
            name='particles')
        self.s_rng = RandomStreams(seed)

    def updates(self, with_s_mu=False):
        new_particles, _locals  = self.rbm.gibbs_step_for_v(
                self.particles,
                self.s_rng,
                return_locals=True)
        if with_s_mu:
            if not hasattr(self.rbm, 's_sample'):
                shp = (self.n_particles,)+self.rbm.s_shp
                self.rbm.s_sample = sharedX(numpy.zeros(shp), 's_sample')
            return {self.particles: new_particles,
                    self.rbm.s_sample: _locals['s_mu']
                    }
        else:
            return {self.particles: new_particles}

    def tile_particles(self, particles=None, shape=None, scale_each=True, **kwargs):
        channel_minor=True
        if shape is None:
            shape = self.rbm.v_shp
            if len(shape)==3 and shape[2]==3:
                cmode = 'RGBA' #now ignored
            elif len(shape)==2:
                cmode = 'L' #now ignored
            if len(shape)==3 and shape[0]==3:
                cmode = 'RGBA' #now ignored
                channel_minor=False
            else:
                raise ValueError('unrecognized colour mode')
        if particles is None:
            particles = self.particles.get_value()
        if not channel_minor:
            if particles.ndim==4:
                assert len(shape)==3
                shape = shape[1], shape[2], shape[0]
                particles = particles.transpose(0,2,3,1) #put colour last
            else:
                raise NotImplementedError()
        return pylearn.io.image_tiling.tile_slices_to_image(
                    particles,
                    scale_each=scale_each,
                    **kwargs)


class Clipper(object):
    """
    A class for bounding a free variable.
    """

    def __init__(self, var, vmin, vmax):
        self.var = var
        self.vmin = vmin
        self.vmax = vmax

    def filter_update(self, ups):
        if self.var in ups:
            ups[self.var] = tensor.clip(ups[self.var],
                    self.vmin,
                    self.vmax)
        return ups


def ndarray_clipper(var, vmin, vmax, logdomain=False, dtype=theano.config.floatX):
    if logdomain:
        vmin = numpy.log(vmin).astype(dtype)
        vmax = numpy.log(vmax).astype(dtype)
    else:
        vmin = numpy.asarray(vmin).astype(dtype)
        vmax = numpy.asarray(vmax).astype(dtype)
    return Clipper(var, vmin, vmax)


class Trainer(object):
    def __init__(self, conf, rbm, sampler, visible_batch, clippers):
        self.conf = conf
        self.rbm = rbm
        self.sampler = sampler
        self.visible_batch = visible_batch
        self.iter=sharedX(0, 'iter')
        self.annealing_coef=sharedX(0.0, 'annealing_coef')
        self.lr_dict = lr_dict = {}
        for p in rbm.params():
            lrname = '%s_lr'%p.name
            lr_dict[p] = sharedX(conf.get(lrname, 1.0), lrname)
        self.clippers = clippers

    def updates(self):
        ups = {}
        add_updates = lambda b: safe_update(ups,b)

        base_lr = numpy.asarray(
                self.conf['base_lr_per_example']/self.conf['batchsize'],
                floatX)
        annealing_coef = clip_ramp(self.iter,
                (self.conf['lr_anneal_start'],base_lr),
                (self.conf['lr_anneal_end'], 0.0),
                dtype=floatX)

        ups[self.iter] = self.iter + 1
        ups[self.annealing_coef] = annealing_coef

        #
        # Enforcing Sparsity
        #
        pos_h = self.rbm.mean_h_given_v(self.visible_batch)
        sparsity_cost = 0
        KL_eps = 1e-4
        if self.conf['sparsity_KL_featuretarget_weight']:
            p = self.conf['sparsity_KL_featuretarget_target']
            sparsity_cost = sparsity_cost + tensor.mul(
                    self.conf['sparsity_KL_featuretarget_weight'],
                    -tensor.sum(
                        p * tensor.log(tensor.mean(pos_h,axis=0)+KL_eps)
                        + (1-p) * tensor.log(1-tensor.mean(pos_h,axis=0)+KL_eps)))
            assert sparsity_cost.ndim==0
            assert sparsity_cost.dtype=='float32'

        if self.conf['sparsity_KL_exampletarget_weight']:
            p = self.conf['sparsity_KL_exampletarget_target']
            sparsity_cost = sparsity_cost + tensor.mul(
                    self.conf['sparsity_KL_exampletarget_weight'],
                    -tensor.sum(
                        p * tensor.log(tensor.mean(pos_h,axis=1)+KL_eps)
                        + (1-p) * tensor.log(1-tensor.mean(pos_h,axis=1)+KL_eps)))
            assert sparsity_cost.ndim==0
            assert sparsity_cost.dtype=='float32'

        #
        # Updates related to CD
        #
        # These updates are for CD-1, PCD/SML, and a stochastic interpolation
        # between them.
        #
        # The idea is to start from negative phase particles neg_v, run them
        # through a step of Gibbs, and put them into sampler.particles:
        #
        #    neg_v -> Gibbs -> sampler.particles.
        #
        # We control the kind of CD by adjusting what neg_v is: either it is the
        # visible_batch (for CD-1) or it is the old sampler.particles (PCD). We
        # can interpolate between the two algorithms by stochastically choosing
        # either the visible_batch[i] or the old particles[i] on a row-by-row
        # basis.
        #
        if self.conf['CD_anneal_start'] < self.conf['CD_anneal_end']:
            P_restart = clip_ramp(self.iter,
                (self.conf['CD_anneal_start'],1.0),
                (self.conf['CD_anneal_end'], 0.0),
                dtype=floatX)
            reset_decisions = self.sampler.s_rng.uniform(
                    size=(self.conf['batchsize'],)) < P_restart
            v0_ndim = self.visible_batch.ndim
            # broadcast reset_decisions over all but batch idx
            neg_v0 = tensor.switch(reset_decisions.dimshuffle(0,*(['x']*(v0_ndim-1))),
                self.visible_batch,       # reset the chain to data
                self.sampler.particles)  # continue old chain
        else:
            neg_v0 = self.sampler.particles
        neg_v1 = self.rbm.gibbs_step_for_v(neg_v0, self.sampler.s_rng)
        ups[self.sampler.particles] = neg_v1

        ## N.B. we are manually advancing the sampler, not calling
        ## Gibbs.updates()
        # add_updates(self.sampler.updates())
        learn_rates = [self.annealing_coef*self.lr_dict[p] for p in self.rbm.params()]
        add_updates(
                self.rbm.cd_updates(
                    pos_v=self.visible_batch,
                    neg_v=neg_v1,
                    lr=learn_rates,
                    other_cost=sparsity_cost))

        #
        # Gathering statistics of unit activity
        #
        neg_h = self.rbm.mean_h_given_v(neg_v1)
        self.pos_h_means = sharedX(numpy.zeros(self.rbm.h_shp)+0.5,'pos_h')
        self.neg_h_means = sharedX(numpy.zeros(self.rbm.h_shp)+0.5,'neg_h')
        ups[self.pos_h_means] = 0.1 * pos_h.mean(axis=0) + .9*self.pos_h_means
        ups[self.neg_h_means] = 0.1 * neg_h.mean(axis=0) + .9*self.neg_h_means

        # Clipping parameters to legal ranges
        for clipper in self.clippers:
            ups = clipper.filter_update(ups)
        return ups

    def assert_is_finite(self):
        particles = self.sampler.particles.get_value(borrow=True)
        assert numpy.all(numpy.isfinite(particles))

    def print_status(self):
        def print_minmax(msg, x):
            print '%s min=%f mean=%f max=%f'%(
                    msg, x.min(), x.mean(), x.max())

        iter = self.iter.get_value()
        print 'iter:', iter, '(%i examples)'% (iter * self.conf['batchsize'])
        self.rbm.print_status()
        print_minmax('particles', self.sampler.particles.get_value(borrow=True))
        print_minmax('pos_h_means', self.pos_h_means.get_value(borrow=True))
        print_minmax('neg_h_means', self.neg_h_means.get_value(borrow=True))
        print 'lr annealing coef:', self.annealing_coef.get_value(borrow=True)

    def get_loop(self, batch_idx):
        return TrainingLoop(batch_idx, self)


class TrainingLoop(object):
    # This is the unsupervised training loop
    # In contrast to the Trial object, this one is *not* intended to be
    # pickled, or even picklable.
    def __init__(self, batch_idx, trainer):
        self.train_fn = theano.function([batch_idx], updates=trainer.updates())
        self.i = int(trainer.iter.get_value())
    def incremental_fit(self, n_iter=1000): #this many minibatches
        i_stop = self.i + n_iter
        while self.i < i_stop:
            self.train_fn(self.i) # the datasets repeat
            self.i += 1           # to match trial.trainer.iter

