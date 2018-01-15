import numpy as np
import tensorflow as tf

from gpflow import settings

from gpflow.decors import autoflow, defer_build, params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.models import Model
from gpflow.params import DataHolder, Minibatch

from .utils import normal_sample, tile_over_samples

class DSDGP(Model):
    """
    Doubly Stochastic Deep Gaussian Process Model.

    Key reference:
        @incollection{NIPS2017_7045,
            title = {Doubly Stochastic Variational Inference for Deep Gaussian
            Processes},
            author = {Salimbeni, Hugh and Deisenroth, Marc},
            booktitle = {Advances in Neural Information Processing Systems 30},
            editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and
            R. Fergus and S. Vishwanathan and R. Garnett},
            pages = {4591--4602},
            year = {2017},
            publisher = {Curran Associates, Inc.},
            url =
            {http://papers.nips.cc/paper/7045-doubly-stochastic-variational-inference-for-deep-gaussian-processes.pdf}
        }
    """
    @defer_build()
    def __init__(self, X, Y, Z, layers, likelihood,
                 num_latent_Y=None,
                 minibatch_size=None,
                 num_samples=1,
                 mean_function=Zero(),
                 name=None):
        """
        - X is a data matrix, size N x D.
        - Y is a data matrix, size N x R.
        - Z is a matrix of inducing inputs, size M x D.
        - layers is an instance of Sequential containing the layer structure of
        the DGP.
        - likelihood is an instance of the gpflow likehood object.
        - num_latent_Y is the number of latent processes to use.
        - minibatch_size, if not None turns of minibatching with that size.
        - num_samples is the number of Monte Carlo samples to use.
        - mean_function is an instance of the gpflow mean_function object,
        corresponds to the mean function of the final layer.
        - name is the name of the TensforFlow object.
        """

        super(DSDGP, self).__init__(name=name)

        assert X.shape[0] == Y.shape[0]
        assert Z.shape[1] == X.shape[1]

        self.num_data, D_X = X.shape
        self.num_samples = num_samples
        self.D_Y = num_latent_Y or Y.shape[1]

        self.layers = layers

        self.dims = self.layers.get_dims()

        self.likelihood = likelihood

        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)

        self.X, self.Y = X, Y

    #Credits to Hugh Salimbeni
    @params_as_tensors
    def _propagate(self, Xnew, full_cov=False, num_samples=1):
        """
        Propagate points Xnew through DGP cascade.
        """
        Fs = [tile_over_samples(Xnew, num_samples), ]
        Fmeans, Fvars = [], []
        for layer in self.layers.layers:
            mean, var = layer._build_predict(Xnew, full_cov, stochastic=True)
            F = normal_sample(mean, var, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[:1], Fmeans, Fvars

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, num_samples=1):
        Fs, Fmeans, Fvars = self._propagate(Xnew, full_cov, num_samples)
        return Fmeans[-1], Fvars[-1]

    #Credits to Hugh Salimbeni
    @params_as_tensors
    def _build_likelihood(self):
        """
        Gives variational bound on the model likelihood.
        """

        Fmean, Fvar = self._build_predict(self.X, full_cov=False, num_samples=self.num_samples)

        Y = tile_over_samples(self.Y, self.num_samples)

        f = lambda a: self.likelihood.variational_expectations(a[0], a[1], a[2])
        var_exp = tf.map_fn(f, (Fmean, Fvar, Y), dtype=settings.float_type)
        var_exp = tf.stack(var_exp) #SN

        var_exp = tf.reduce_mean(var_exp, 0) # S,N -> N. Average over samples
        L = tf.reduce_sum(var_exp) # N -> scalar. Sum over data (minibatch)

        KL = 0.
        for layer in self.layers.layers:
            KL += layer.build_prior_KL(K=None)

        scale = tf.cast(self.num_data, settings.float_type)
        scale /= tf.cast(tf.shape(self.X)[0], settings.float_type)  # minibatch size
        return L * scale - KL

    #Credits to gpflow dev team
    @autoflow((settings.float_type, [None, None]), (tf.int32, []))
    def predict_f(self, Xnew, num_samples):
        """
        Compute the mean and variance of the latent function(s) for the final
        layer at the points Xnew.
        """
        return self._build_predict(Xnew, num_samples)

    #Credits to gpflow dev team
    @autoflow((settings.float_type, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) for
        the final layer at the points Xnew.
        """
        return self._build_predict(Xnew, full_cov=True)


    #Credits to Hugh Salimbeni
    @autoflow((settings.float_type, [None, None]), (tf.int32, []))
    def predict_all_layers(self, Xnew, num_samples):
        """
        Compute the mean and variance of the latent function(s) for for all
        layers at the points Xnew.
        """
        return self._propagate(Xnew, num_samples)

    #Credits to Hugh Salimbeni
    @autoflow((settings.float_type, [None, None]))
    def predict_all_layers_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) for
        all layers at the points Xnew.
        """
        return self._propagate(Xnew, full_cov=True)

    #Credits to gpflow dev team
    @autoflow((settings.float_type, [None, None]), (tf.int32, []))
    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) for the final
        layer at the points Xnew.
        """
        mu, var = self._build_predict(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=settings.float_type) * settings.numerics.jitter_level
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i] + jitter)
            shape = tf.stack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=settings.float_type)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.stack(samples))

    #Credits to gpflow dev team
    @autoflow((settings.float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    #Credits to gpflow dev team
    # @autoflow((settings.float_type, [None, None]), (settings.float_type, [None, None]))
    # def predict_density(self, Xnew, Ynew):
        # """
        # Compute the (log) density of the data Ynew at the points Xnew
        # Note that this computes the log density of the data individually,
        # ignoring correlations between them. The result is a matrix the same
        # shape as Ynew containing the log densities.
        # """
        # pred_f_mean, pred_f_var = self._build_predict(Xnew)
        # return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)
