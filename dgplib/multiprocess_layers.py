from typing import Tuple

import abc
import tensorflow as tf

from gpflow.base import Module

from .layers import Layer


class MultiprocessLayer(Module):
    """
    Inherits from Layer class. Can handle outputs from different priors.
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

        sublayers = []
        for i in range(self.num_sublayers):
            sublayer = Layer(input_dim=self.input_dim, output_dim=self.sublayer_output_dim, kernel=kernels[i],
                             feature=feature, num_inducing=num_inducing, share_Z=share_Z,
                             fixed_linear_mean_function=fixed_linear_mean_function, mean_function=mean_functions[i],
                             whiten=whiten, q_diag=q_diag, q_mu=q_mu, q_sqrt=q_sqrt)
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
    def predict_f(self, Xnew: tf.Tensor, full_cov=False, full_output_cov=False) -> Tuple[tf.Tensor, tf.Tensor]:
        pass

    @abc.abstractmethod
    def predict_f_samples(self, Xnew: tf.Tensor, num_samples=1, full_cov=False, full_output_cov=False) -> tf.Tensor:
        pass

    def propagate_inputs_and_features(self, X, Z):
        """
        Returns an initialization for the data and inducing inputs for the consequent layer
        :param X: inputs
        :param Z: inducing inputs
        """
        return self.sublayers[0].propagate_inputs_and_features(X, Z)

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


class ConcatinativeMultiprocessLayer(MultiprocessLayer):
    @property
    def output_dim(self):
        self.sublayer_output_dim * self.num_sublayers
