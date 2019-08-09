from typing import Tuple

import tensorflow as tf
import numpy as np

from gpflow.base import Parameter, Module, positive, triangular
from gpflow.conditionals import conditional, sample_conditional
from gpflow.config import default_float, default_jitter
from gpflow.covariances import Kuu
from gpflow.features import Mof, SeparateIndependentMof, SharedIndependentMof, InducingPoints
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Linear, Zero

from .utilities import find_linear_mf_weights


class Layer(Module):
    """
    The basic SVGP layer class. Handles input_dim and output_dim.
    """

    def __init__(self, input_dim, output_dim, kernel, feature=None,
                 num_inducing=None, share_Z=False, fixed_linear_mean_function=False,
                 mean_function=None, whiten=True, q_diag=False, q_mu=None, q_sqrt=None, name=None):
        """
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param kernel: GPflow kernel
        :param feature: numpy array of inducing inputs or InducingFeature object or None
        :param num_inducing: number of inducing inputs, automatically infered if feature is not None
        :param share_Z: True if inducing inputs are shared across layer dimensions
        :param fixed_linear_mean_function: True of mean function is fixed
         like in Salimbeni and Diesenroth 2017
        :param mean_function: GPflow mean_function
        :param whiten: True if whitened representation is used
        :param q_diag: True if variational covariance is set to diagonal
        :param q_mu: numpy array of variational mean
        :param q_sqrt: numpy array of Cholesky of variational covariance
        """

        super().__init__(name=name)

        self.input_dim = input_dim
        self.output_dim = output_dim

        if feature is not None:
            assert isinstance(feature, Mof)
        self.feature = feature

        self.kernel = kernel
        self.share_Z = share_Z

        self.fixed_linear_mean_function = fixed_linear_mean_function
        if self.fixed_linear_mean_function:
            self.mean_function = Linear()
            self.mean_function.trainable = False
        else:
            self.mean_function = mean_function or Zero(output_dim=self.output_dim)

        self.whiten = whiten

        if self.feature is not None:
            num_inducing = len(self.feature)

        self.q_diag = q_diag
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Modification from GPflow.
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.
        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.
        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros(
            (num_inducing, self.output_dim)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.output_dim),
                               dtype=default_float())
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_sqrt = [
                    np.eye(num_inducing, dtype=default_float())
                    for _ in range(self.output_dim)
                ]
                q_sqrt = np.array(q_sqrt)
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                assert self.output_dim == q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                assert self.output_dim == q_sqrt.shape[0]
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    def prior_kl(self):
        """
        KL(q(u)||p(u))
        """
        K = None
        if not self.whiten:
            K = Kuu(self.feature, self.kernel, jitter=default_jitter())  # [P, M, M] or [M, M]

        return gauss_kl(self.q_mu, self.q_sqrt, K)

    def predict_f(self, Xnew: tf.Tensor, full_cov=False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Returns the posterior mean and covariance at Xnew

        :param Xnew: input (tf.Tensor or numpy array)
        :param full_cov: True if full covariance required
        """
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = conditional(
            Xnew,
            self.feature,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=False
        )

        return mu + self.mean_function(Xnew), var

    def predict_f_samples(self, Xnew: tf.Tensor, num_samples=1, full_cov=False) -> tf.Tensor:
        """
        Returns sample from GP posterior at Xnew

        :param Xnew: input (tf.Tensor or numpy array)
        :param num_samples: number of MC samples
        :param full_cov: True if full covariance required
        """
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        samples, mu, var = sample_conditional(
            Xnew,
            self.feature,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=False,
            num_samples=num_samples
        )

        return samples + self.mean_function(Xnew), mu + self.mean_function(Xnew), var

    def propagate_inputs_and_features(self, X, Z):
        """
        Returns an initialization for the data and inducing inputs for the consequent layer
        :param X: inputs
        :param Z: inducing inputs
        """
        W = find_linear_mf_weights(self.input_dim, self.output_dim, X)

        Z_running = Z.copy().dot(W)
        X_running = X.copy().dot(W)

        if X.shape[1] - self.input_dim == 1:
            Z_running = np.hstack([Z_running, Z[:, -1:]])
            X_running = np.hstack([X_running, X[:, -1:]])

        return X_running, Z_running, W

    def initialize_features(self, Z):
        """
        Initialize the inducing inputs/features for this Layer
        :param Z: inducing input values
        """
        if self.feature is None:
            if self.share_Z:
                self.feature = SharedIndependentMof(InducingPoints(Z))
            else:
                self.feature = SeparateIndependentMof([InducingPoints(Z) for _ in range(self.output_dim)])
        else:
            raise ValueError("Features already initialized")

    def initialize_linear_mean_function_weights(self, W):
        """
        Initialize linear mean function weights for this Layer
        :param W: numpy array of linear mean function weights
        """
        if self.fixed_linear_mean_function:
            self.mean_function.A = Parameter(W)
            self.mean_function.A.trainable = False
            self.mean_function.b.trainable = False
        else:
            raise ValueError("Mean function is not specified as fixed on construction.")
