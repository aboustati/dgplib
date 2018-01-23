from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from gpflow import settings

from gpflow.conditionals import conditional
from gpflow.decors import params_as_tensors, autoflow, defer_build
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Linear, Zero
from gpflow.params import Parameter, Parameterized, ParamList

from .layers import Layer, find_weights
from .utils import shape_as_list

class MultikernelLayer(Layer):
    """
    Inherits from Layer class. Can handle outputs from different priors.
    """

    @defer_build()
    def __init__(self, input_dim, output_dim, num_inducing, kernel_list,
                 share_Z=False, mean_function=None, name=None):

        if output_dim != len(kernel_list):
            raise ValueError("Number of kernels must match output dimension")

        super(Layer, self).__init__(input_dim=input_dim,
                                    output_dim=output_dim,
                                    num_inducing=num_inducing,
                                    kernel=kernel_list,
                                    mean_function=mean_function,
                                    name=name)

        self.num_kernels = len(kernel_list)
        self._shared_Z = share_Z

        if not self._shared_Z:
            del self.Z
            Z = Parameter(np.zeros((self.num_inducing, self.input_dim)), fix_shape=True)
            self.Z = ParamList([Z.copy() for _ in range(self.num_kernels)])


    @params_as_tensors
    def build_prior_KL(self, K):
        if K:
            KL = 0.
            for i, k in enumerate(K):
                KL += gauss_kl_white(self.q_mu[:,i][:,None],
                                     self.q_sqrt[i,:,:][None,:,:],
                                     K=k
                                    )
            return KL
        else:
            return gauss_kl(self.q_mu, self.q_sqrt, K=K)

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, stochastic=True):
        def f_conditional(Xnew, full_cov=False):
            mean = []
            var = []
            if self._shared_Z:
                Zs = [self.Z for _ in range(self.num_kernels)]
            else:
                Zs = self.Z
            for i, (k, Z) in enumerate(zip(self.kernel, Zs)):
            m, v = conditional(Xnew=Xnew,
                               X=Z,
                               kern=k,
                               f=self.q_mu[:,i][:,None],
                               q_sqrt=self.q_sqrt[i,:,:,][None,:,:],
                               full_cov=full_cov,
                               white=True)

            mean = tf.stack(mean, axis=-1) #NxK
            var = tf.stack(var, axis=-1) #NxK or NxNxK

            return mean + self.mean_function(Xnew), var

        def multisample_conditional(Xnew, full_cov=False):
            if full_cov:
                f = lambda a: f_conditional(a, full_cov=full_cov)
                mean, var = tf.map_fn(f, Xnew, dtype=(settings.tf_float,
                                                  settings.tf_float))
                return tf.stack(mean), tf.stack(var)
            else:
                #S, N, D = shape_as_list(Xnew)
                s = tf.shape(Xnew)
                X_flat = tf.reshape(Xnew, [s[0]*s[1], s[2]])
                mean, var = f_conditional(X_flat)
                return [tf.reshape(m, [s[0], s[1], -1]) for m in [mean, var]]

        if stochastic:
            mean, var = multisample_conditional(Xnew, full_cov)
        else:
            mean, var = conditional(Xnew, full_cov)

        return mean, var

class MultikernelInputLayer(MultikernelLayer):
    @defer_build()
    def __init__(self, input_dim, output_dim, num_inducing, kernel_list,
                 mean_function=None, name=None):
        """
        input_dim is an integer
        output_dim is an integer
        num_inducing is the number of inducing inputs
        kernel is a kernel object (or list of kernel objects)
        """

        super(MultikernelInputLayer, self).__init__(input_dim, output_dim, num_inducing,
                                         kernel_list, mean_function, name)


    @defer_build()
    def initialize_forward(self, X, Z):
        """
        Initialize Layer and Propagate values of inputs and inducing inputs
        forward
        """

        W = find_weights(self.input_dim, self.output_dim, X)

        if self._shared_Z:
            self.Z.assign(Z)
        else:
            for Z_current in self.Z:
                Z_current.assign(Z)

        Z_running = Z.copy().dot(W)
        X_running = X.copy().dot(W)

        if isinstance(self.mean_function, Linear):
            self.mean_function.A = W

        return X_running, Z_running


class MultikernelHiddenLayer(MultikernelLayer):
    @defer_build()
    def __init__(self, input_dim, output_dim, num_inducing, kernel_list,
                 mean_function=None, name=None):
        """
        input_dim is an integer
        output_dim is an integer
        num_inducing is the number of inducing inputs
        kernel_list is list of kernel objects
        """

        super(MultikernelHiddenLayer, self).__init__(input_dim,
                                                     output_dim,
                                                     num_inducing,
                                                     kernel_list,
                                                     mean_function,
                                                     name)

    @defer_build()
    def initialize_forward(self, X, Z):
        """
        Initialize Layer and Propagate values of inputs and inducing inputs
        forward
        """

        W = find_weights(self.input_dim, self.output_dim, X)

        if self._shared_Z:
            self.Z.assign(Z)
        else:
            for Z_current in self.Z:
                Z_current.assign(Z)

        Z_running = Z.copy().dot(W)
        X_running = X.copy().dot(W)

        if isinstance(self.mean_function, Linear):
            self.mean_function.A =W

        return X_running, Z_running

class MultikernelOutputLayer(Layer):
    @defer_build()
    def initialize_forward(self, X, Z):
        """
        Initialize Layer and Propagate values of inputs and inducing inputs
        forward
        """

        if self._shared_Z:
            self.Z.assign(Z)
        else:
            for Z_current in self.Z:
                Z_current.assign(Z)

        return (None, None)
