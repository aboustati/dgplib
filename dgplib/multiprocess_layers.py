from typing import Tuple

import abc
import numpy as np
import tensorflow as tf

from gpflow.base import Module

from .layers import Layer


class MultiprocessLayer(Module):
    """
    Abstract class defining multiprocess layer construction.
    """

    def __init__(self, input_dim, sublayer_output_dim, kernels, feature=None,
                 num_inducing=None, share_Z=False, fixed_linear_mean_function=False,
                 mean_functions=None, whiten=True, q_diag=False, q_mu=None, q_sqrt=None, name=None):
        super().__init__(name=name)

        self.input_dim = input_dim
        self.sublayer_output_dim = sublayer_output_dim
        self.num_sublayers = len(kernels)

        if mean_functions is None:
            mean_functions = [None] * self.num_sublayers

        self.fixed_linear_mean_function = fixed_linear_mean_function

        sublayers = []
        for i in range(self.num_sublayers):
            sublayer = Layer(input_dim=self.input_dim, output_dim=self.sublayer_output_dim, kernel=kernels[i],
                             feature=feature, num_inducing=num_inducing, share_Z=share_Z,
                             fixed_linear_mean_function=self.fixed_linear_mean_function,
                             mean_function=mean_functions[i], whiten=whiten, q_diag=q_diag, q_mu=q_mu, q_sqrt=q_sqrt)
            sublayers.append(sublayer)

        self.sublayers = sublayers

    @property
    @abc.abstractmethod
    def output_dim(self):
        pass

    def prior_kl(self):
        KL = 0.
        for sublayer in self.sublayers:
            KL += sublayer.prior_kl()
        return KL

    @abc.abstractmethod
    def predict_f(self, Xnew: tf.Tensor, full_cov=False) -> Tuple[tf.Tensor, tf.Tensor]:
        pass

    @abc.abstractmethod
    def predict_f_samples(self, Xnew: tf.Tensor, num_samples=1, full_cov=False) -> tf.Tensor:
        pass

    @abc.abstractmethod
    def propagate_inputs_and_features(self, X, Z):
        """
        Returns an initialization for the data and inducing inputs for the consequent layer
        :param X: inputs
        :param Z: inducing inputs
        """
        pass

    def initialize_features(self, Z):
        """
        Initialize the inducing inputs/features for this Layer
        :param Z: inducing input values
        """
        for sublayer in self.sublayers:
            sublayer.initialize_features(Z)

    def initialize_linear_mean_function_weights(self, W):
        """
        Initialize linear mean function weights for this Layer
        :param W: numpy array of linear mean function weights
        """
        for sublayer in self.sublayers:
            sublayer.initialize_linear_mean_function_weights(W)


class ConcatinativeMultiprocessLayerMixin:
    @property
    def output_dim(self):
        return self.sublayer_output_dim * self.num_sublayers

    def propagate_inputs_and_features(self, X, Z):
        """
        Returns an initialization for the data and inducing inputs for the consequent layer
        :param X: inputs
        :param Z: inducing inputs
        """
        X_running, Z_running = [], []
        for sublayer in self.sublayers:
            X_sub, Z_sub, W = sublayer.propagate_inputs_and_features(X, Z)  # NxD_sub(+1), MxD_sub(+1), D_in(+1)xD_sub
            X_running.append(X_sub)
            Z_running.append(Z_sub)

        # Hack to make propagation of index column possible
        if X.shape[1] - self.input_dim == 1:
            X_running = np.hstack([xx[:, :-1] for xx in X_running] + [X[:, -1:]])
            Z_running = np.hstack([zz[:, :-1] for zz in Z_running] + [Z[:, -1:]])
        else:
            X_running = np.hstack(X_running)
            Z_running = np.hstack(Z_running)

        return X_running, Z_running, W

    def predict_f(self, Xnew: tf.Tensor, full_cov=False):
        mu, var = [], []
        for sublayer in self.sublayers:
            m, v = sublayer.predict_f(Xnew=Xnew, full_cov=full_cov)
            mu.append(m)
            var.append(v)
        mu = tf.concat(mu, axis=-1)
        if full_cov:
            var = tf.concat(var, axis=0)
        else:
            var = tf.concat(var, axis=-1)

        return mu, var

    def predict_f_samples(self, Xnew: tf.Tensor, num_samples=1, full_cov=False):
        samples, mu, var = [], [], []
        for sublayer in self.sublayers:
            s, m, v = sublayer.predict_f_samples(Xnew=Xnew, num_samples=num_samples, full_cov=full_cov)
            samples.append(s)
            mu.append(m)
            var.append(v)
        samples = tf.concat(samples, axis=-1)
        mu = tf.concat(mu, axis=-1)
        if full_cov:
            var = tf.concat(var, axis=1)
        else:
            var = tf.concat(var, axis=-1)

        return samples, mu, var


class AdditiveMultiprocessLayerMixin:
    @property
    def output_dim(self):
        return self.sublayer_output_dim

    def propagate_inputs_and_features(self, X, Z):
        """
        Returns an initialization for the data and inducing inputs for the consequent layer
        :param X: inputs
        :param Z: inducing inputs
        """
        X_running, Z_running = [], []
        for sublayer in self.sublayers:
            X_sub, Z_sub, W = sublayer.propagate_inputs_and_features(X, Z)  # NxD_sub(+1), MxD_sub(+1), D_in(+1)xD_sub
            X_running.append(X_sub)
            Z_running.append(Z_sub)

        X_running = np.sum(np.stack(X_running, axis=0), axis=0)
        Z_running = np.sum(np.stack(Z_running, axis=0), axis=0)
        if X.shape[1] - self.input_dim == 1:
            X_running = np.hstack([X_running, X[:, -1:]])
            Z_running = np.hstack([Z_running, Z[:, -1:]])

        return X_running, Z_running, W

    def predict_f(self, Xnew: tf.Tensor, full_cov=False):
        mu, var = [], []
        for sublayer in self.sublayers:
            m, v = sublayer.predict_f(Xnew=Xnew, full_cov=full_cov)
            mu.append(m)
            var.append(v)
        mu = tf.reduce_sum(tf.stack(mu), axis=0)
        var = tf.reduce_sum(tf.stack(var), axis=0)

        return mu, var

    def predict_f_samples(self, Xnew: tf.Tensor, num_samples=1, full_cov=False):
        samples, mu, var = [], [], []
        for sublayer in self.sublayers:
            s, m, v = sublayer.predict_f_samples(Xnew=Xnew, num_samples=num_samples,  full_cov=full_cov)
            samples.append(s)
            mu.append(m)
            var.append(v)
        samples = tf.reduce_sum(tf.stack(samples), axis=0)
        mu = tf.concat(mu, axis=0)
        var = tf.reduce_sum(tf.stack(var), axis=0)

        return samples, mu, var


class ConcatinativeMultiprocessLayer(ConcatinativeMultiprocessLayerMixin, MultiprocessLayer):
    pass

class AdditiveMultiprocessLayer(AdditiveMultiprocessLayerMixin, MultiprocessLayer):
    pass

