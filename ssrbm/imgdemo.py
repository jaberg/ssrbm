"""
Demonstrate learning an ssRBM model of tinyimages
"""
import cPickle
import sys
import logging
if __name__=='__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger('kriz')
logger.setLevel(logging.INFO)

import numpy

import theano
from theano import function, tensor, shared

import skdata.cifar10
import theano_linear

from utils import sharedX, floatX
import rbm


def preprocess(batch_x, dct, x_shape):
    if dct['kind'] == 'RAW':
        visible_pos_phase = batch_x
    elif dct['kind'] == 'GCN':
        x2 = tensor.cast(batch_x.flatten(2), 'float32')
        x2c = x2 - x2.mean(axis=1).dimshuffle(0,'x')
        x2lcn = x2c / tensor.sqrt((10 + (x2c**2).mean(axis=1).dimshuffle(0,'x')))
        visible_pos_phase = x2lcn.reshape(x_shape)
    else:
        raise NotImplementedError(dct['kind'])
    return visible_pos_phase


class TrialObject(object):
    def __init__(self, conf):
        self.conf = conf

        batchsize = conf['batchsize']
        rng = numpy.random.RandomState(conf['seed'])

        batch_x = tensor.ftensor4()
        # assume it's (images, rows, columns, channels)

        visible_pos_phase = preprocess(batch_x, conf['preprocessing'],
                x_shape = (batchsize, 32, 32, 3))

        filter_shape = conf['filter_shape']
        n_filters = conf['n_filters']
        W = filters = global_weights = None
        if 1: #always allocate W
            if conf['n_filters']:
                filters = theano_linear.LConv(
                        filters=sharedX(rng.randn(*((n_filters,3,)+filter_shape)) *
                            conf['filters_irange'] , 'filters'),
                        img_shape=(batchsize,3,32,32))
            W = filters
            if conf['n_global_weights']:
                global_weights = theano_linear.MatrixMul(
                        W=sharedX(rng.randn(32*32*3, conf['n_global_weights']) * conf['global_weights_irange'],
                            'global_weights'),
                        row_shape=(3,32,32))
            W = theano_linear.LConcat([global_weights, W]) if W else global_weights

        Lambda = Lambda_filters = Lambda_weights = None
        if conf['Lambda_enabled']: # Allocate Lambda_weights, Lambda_filters
            if conf['n_filters']:
                if conf['Lambda_filters_logdomain']: raise NotImplementedError()
                Lambda = Lambda_filters = theano_linear.LConv(
                        filters=sharedX(
                            numpy.add(
                                numpy.zeros(((n_filters,3,)+filter_shape)),
                                conf['Lambda_filters0']),
                            'Lambda_filters'),
                        img_shape=(batchsize,3,32,32))
            if conf['n_global_weights']:
                if conf['Lambda_weights_logdomain']: raise NotImplementedError()
                Lambda_weights = theano_linear.MatrixMul(
                        W=sharedX(
                            numpy.add(
                                numpy.zeros((3*32*32, conf['n_global_weights'])),
                                conf['Lambda_weights0']),
                            'Lambda_weights'),
                        row_shape=(3,32,32))
            Lambda = theano_linear.LConcat([Lambda_weights, Lambda]) if Lambda else Lambda_weights

        assert 0
        rbm = template.RBM(conf, rng, W, Lambda=Lambda)

        sampler = template.Gibbs(rbm,
                particles=rng.randn(batchsize, *rbm.v_shp),
                rng=rng)

        #
        # Set up Clipping objects for trainer
        #

        clippers = []
        clippers.append( template.ndarray_clipper( rbm.alpha,
                    conf['alpha_min'],
                    conf['alpha_max'],
                    conf['alpha_logdomain']))
        clippers.append( template.ndarray_clipper( rbm.B,
                    conf['B_min'],
                    conf['B_max']))
        if Lambda_filters:
            clippers.append(template.ndarray_clipper(Lambda_filters._filters,
                        conf['Lambda_filters_min'],
                        conf['Lambda_filters_max'],
                        conf['Lambda_filters_logdomain']))
        if Lambda_weights:
            clippers.append(template.ndarray_clipper(Lambda_weights._W,
                        conf['Lambda_weights_min'],
                        conf['Lambda_weights_max'],
                        conf['Lambda_weights_logdomain']))
        clippers.append(template.ndarray_clipper(sampler.particles,
                    conf['particles_min'],
                    conf['particles_max']))

        #
        # Allocate trainer
        #

        trainer = template.Trainer(conf, rbm, sampler,
                visible_batch=visible_pos_phase,
                clippers=clippers)

        self.rbm = rbm
        self.sampler = sampler
        self.trainer = trainer
        self.batch_x = batch_x
        self.visible_pos_phase = visible_pos_phase
        self.Lambda_weights = Lambda_weights
        self.Lambda_filters = Lambda_filters
        self.W = W
        self.filters = filters
        self.global_weights = global_weights

    def save(self, filename, version=-1):
        f = open(filename, 'wb')
        cPickle.dump(self, f, version)
        f.close()


class UnsupervisedTraining(object):
    # This is the unsupervised training loop
    # In contrast to the Trial object, this one is *not* intended to be
    # pickled, or even picklable.
    def __init__(self, trial):
        self.trial = trial
        self.input_fn = theano.function(
                [trial.batch_idx],
                [trial.batch_x, trial.visible_pos_phase])
        self.train_fn = theano.function(
                [trial.batch_idx],
                updates=trial.trainer.updates())
        self.i = int(trial.trainer.iter.get_value())

    def incremental_fit(self, n_iter=1000): #this many minibatches
        trial = self.trial
        rbm = self.trial.rbm
        sampler = self.trial.sampler
        trainer = self.trial.trainer
        i_stop = self.i + n_iter
        while self.i < i_stop:
            self.train_fn(self.i) # the datasets repeat
            self.i += 1      # to match trial.trainer.iter
            print_status=0
            if self.i <= 500 and self.i % 50 == 0:
                print_status=1
            elif self.i % 500 == 0:
                print_status = 1
            if print_status:
                print ''
                trainer.print_status()
                if trial.global_weights:
                    trial.global_weights.tile_columns(channel_major=True,
                            scale_each=True).save('global_weights_%06i.png'%self.i)
                    trial.global_weights.print_status()
                if trial.filters:
                    trial.filters.tile_columns(scale_each=True).save('filters_%06i.png'%self.i)
                    trial.filters.print_status()
                if trial.Lambda_weights:
                    trial.Lambda_weights.tile_columns(channel_major=True,
                            scale_each=True).save('Lambda_weights_%06i.png'%self.i)
                    trial.Lambda_weights.print_status()
                if trial.Lambda_filters:
                    trial.Lambda_filters.tile_columns(scale_each=True).save('Lambda_filters_%06i.png'%self.i)
                    trial.Lambda_filters.print_status()
                sampler.tile_particles(scale_each=False).save('particles_%06i.png'%self.i)

                if self.i < 500:
                    orig_x, preproc_x = self.input_fn(self.i)
                    sampler.tile_particles(particles=orig_x,scale_each=False).save('input_%06i_a.png'%self.i)
                    sampler.tile_particles(particles=preproc_x,scale_each=False).save('input_%06i_b.png'%self.i)


def conf_init(conf):
    conf.update(dict(
        lr_anneal_start=20000,
        visible_space='pixels',
        seed=3234,
        base_lr_per_example=0.0003,
        batchsize=128,
        borderwidth=0,
        n_filters=64,
        filter_shape=(9,9),
        filters_lr=.01, #relative to base
        filters_irange=0.01,
        n_global_weights=1000,
        global_weights_irange=.01,
        alpha0=10.0,
        alpha_irange=0.0,
        alpha_logdomain=True,
        alpha_min=1.0,
        alpha_max=100.0,
        Lambda_enabled=False,
        mu0=1.0,
        mu_irange=0.0, #TODO: implement this
        b0=0.0,
        b_irange=0.0,
        B0=10.0,
        B_min=5.0, #permit most of v precision to go into Lambda
        B_max=1000.0,
        B_full_diag=True,
        sparsity_KL_featuretarget_weight=0,
        sparsity_KL_exampletarget_weight=0,
        apply_sparsity_to_everything=0,
        particles_min=-50,
        particles_max=50,
        ))
    return conf


def do_lcn(conf):
    conf['preprocessing'] = dict(
            kind='LCN',
            width=9)
    return conf

def do_gcn(conf):
    conf['preprocessing'] = dict(kind='GCN')
    return conf

def with_Lambda(conf):
    conf.update(dict(
        Lambda_enabled=True,
        Lambda_filters0 = 0.00001,
        Lambda_filters_min = 0.00001,
        Lambda_filters_max = 10.0,
        Lambda_filters_lr = conf['filters_lr'],
        Lambda_filters_logdomain = False,
        Lambda_weights0 = 0.00001,
        Lambda_weights_min = 0.00001,
        Lambda_weights_max = 10.0,
        Lambda_weights_lr = 1.0,
        Lambda_weights_logdomain = False,
        ))
    return conf


def with_momentum(conf):
    # make sure to update Pylearn to get the new sgd_momentum_updates fn
    conf['momentum'] = .9
    return conf


def conf_print(dct):
    keys = dct.keys()
    keys.sort()
    for k in keys:
        logger.info('%s=%s'%(k, dct[k]))


# Call like this:
# python imgdemo.py main_train
# ~/cvs/Pycall/bin/pycall 'ssrbm.Tssrbm.kriz.main("do_lcn", dataset="cifar10", base_lr_per_example=3e-4)'
def main_train():
    conf = conf_init(dict())

    # TODO: for initializer in initializers: conf = initializer(conf)
    conf = do_gcn(conf)
    #conf = with_Lambda(conf)
    #conf = with_momentum(conf)

    conf_print(conf)

    trial = TrialObject(conf)
    loop = UnsupervisedTraining(trial)
    trial.save('model_%06i.pkl'%loop.i)
    for i in range(10):
        loop.incremental_fit(1000)
        trial.save('model_%06i.pkl'%loop.i)
    for i in range(9):
        loop.incremental_fit(10000)
        trial.save('model_%06i.pkl'%loop.i)


def main_continue_cd(trial_filename, N):
    trial = cPickle.load(open(trial_filename))
    loop = UnsupervisedTraining(trial)
    loop.incremental_fit(int(N))
    trial.save('model_continued_%i.pkl'%loop.i)


def main_sample(trial_filename):
    saved_trial = cPickle.load(open(trial_filename))

    # WARNING: This is trying to work-around a bug that seems to screw up
    # unpickling... setting the annealing_coef.set_value(0) does not seem to
    # stop learning.
    # The workaround here is to rebuild the Trial object from scratch
    new_conf = dict(saved_trial.conf)
    new_conf['CD_anneal_start'] = -1
    new_conf['CD_anneal_end'] = -1
    new_conf['seed'] += 1 # so things initialize differently

    new_trial = TrialObject(new_conf)
    loop = UnsupervisedTraining(new_trial)

    # Copy over all important state information
    for new_p, old_p in zip(new_trial.rbm.params(), saved_trial.rbm.params()):
        new_p.set_value(old_p.get_value())
    new_trial.sampler.particles.set_value(
            saved_trial.sampler.particles.get_value())

    # cool learning rate to zero
    anneal_T = 10000
    trainer = new_trial.trainer
    sampler = new_trial.sampler

    anneal0 = saved_trial.trainer.annealing_coef.get_value()
    print 'cooling model for %i iters'%anneal_T
    for i in range(anneal_T):
        trainer.annealing_coef.set_value(
                numpy.float32(anneal0 * (anneal_T-i) / float(anneal_T)))
        loop.train_fn(i + loop.i)
        sys.stdout.write('.')
        sys.stdout.flush()

    burnin = 0
    print 'burning in / sampling'
    sampler.tile_particles(scale_each=False).save(
            '%s_sample_%06i_c.png'%(trial_filename[:-4],burnin))
    while True:
        for i in xrange(1000):
            trainer.annealing_coef.set_value(0)
            loop.train_fn(0)
            burnin += 1
        print burnin
        sampler.tile_particles(scale_each=False).save(
                '%s_sample_%06i_c.png'%(trial_filename[:-4],burnin))

if __name__ == '__main__':
    main = globals()[sys.argv[1]]
    sys.exit(main(*sys.argv[2:]))
