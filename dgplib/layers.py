from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from gpflow import settings

from gpflow.conditionals import conditional
from gpflow.decors import params_as_tensors, autoflow, defer_build
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Linear, Zero
from gpflow.params import Parameter, Parameterized, ParamList

from .utils import shape_as_list

class Layer(Parameterized):
    """
    The basic layer class. Handles input_dim and output_dim.
    """

    @defer_build()
    def __init__(self, input_dim, output_dim, num_inducing, kernel,
                 mean_function=None, name=None):
        """
        input_dim is an integer
        output_dim is an integer
        num_inducing is the number of inducing inputs
        kernel is a kernel object (or list of kernel objects)
        """

        super(Layer, self).__init__(name=name)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = num_inducing
        self.Z = Parameter(np.zeros((self.num_inducing, self.input_dim)),
                           fix_shape=False)

        if isinstance(kernel, list):
            self.kernel = ParamList(kernel)
        else:
            self.kernel = kernel

        self.mean_function = mean_function or Zero()

        shape = (self.num_inducing, self.output_dim)

        self.q_mu = Parameter(np.zeros(shape))

        q_sqrt = np.dstack([np.eye(self.num_inducing)
                            for _ in range(self.output_dim)])
        self.q_sqrt = Parameter(q_sqrt)

    @params_as_tensors
    def build_prior_KL(self, K):
        return gauss_kl(self.q_mu, self.q_sqrt, K=K)

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, stochastic=True):
        # Credits to High Salimbeni for this (@hughsalimbeni)
        def f_conditional(Xnew, full_cov=False):
            mean, var = conditional(Xnew=Xnew,
                                    X=self.Z,
                                    kern=self.kern,
                                    f=self.q_mu,
                                    q_sqrt=self.q_sqrt,
                                    full_cov=full_cov,
                                    white=True)

            return mean + self.mean_function(Xnew), var

        def multisample_conditional(Xnew, full_cov=False):
            if full_cov:
                f = lambda a: f_conditional(a, full_cov=full_cov)
                mean, var = tf.map_fn(f, Xnew, dtype=(settings.tf_float,
                                                  settings.tf_float))
                return tf.stack(mean), tf.stack(var)
            else:
                S, N, D = shape_as_list(Xnew)
                X_flat = tf.reshape(Xnew, [S*N, D])
                mean, var = f_conditional(X_flat)
                return [tf.reshape(m, [S, N, -1]) for m in [mean, var]]

        if stochastic:
            mean, var = multisample_conditional(Xnew, full_cov)
        else:
            mean, var = conditional(Xnew, full_cov)

        return mean, var

def find_weights(input_dim, output_dim, X):
    """
    Find the initial weights of the Linear mean function based on
    input and output dimensions of the layer
    """

    if input_dim == output_dim:
        W = np.eye(input_dim)
    elif input_dim > output_dim:
        _, _, V = np.linalg.svd(X, full_matrices=False)
        W = V[:output_dim, :].T
    elif input_dim < output_dim:
        I = np.eye(input_dim)
        zeros = np.zeros((input_dim, output_dim - input_dim))
        W = np.concatenate([I, zeros], 1)

    return W

class InputLayer(Layer):
    @defer_build()
    def __init__(self, input_dim, output_dim, Z, num_inducing, kernel,
                 mean_function=None, name=None):
        """
        input_dim is an integer
        output_dim is an integer
        Z is a matrix of inducing inputs
        num_inducing is the number of inducing inputs
        kernel is a kernel object (or list of kernel objects)
        """

        super(InputLayer, self).__init__(input_dim, output_dim, num_inducing,
                                         kernel, mean_function, name)

        assert Z.shape[0] == self.num_inducing
        assert Z.shape[1] == self.input_dim

        self.Z.assign(Z)


    @defer_build()
    def initialize_forward(self, X):
        """
        Initialize Layer and Propagate values of inputs and inducing inputs
        forward
        """

        W = find_weights(self.input_dim, self.output_dim, X)

        Z_running = self.Z.value.copy().dot(W)
        X_running = X.copy().dot(W)

        if isinstance(self.mean_function, Linear):
            self.mean_function.A = W

        return Z_running, X_running


class HiddenLayer(Layer):
    @defer_build()
    def __init__(self, input_dim, output_dim, num_inducing, kernel,
                 mean_function=None, name=None):
        """
        input_dim is an integer
        output_dim is an integer
        num_inducing is the number of inducing inputs
        kernel is a kernel object (or list of kernel objects)
        """

        super(HiddenLayer, self).__init__(input_dim, output_dim, num_inducing, kernel,
                                          mean_function, name)

    @defer_build()
    def initialize_forward(self, X, Z):
        """
        Initialize Layer and Propagate values of inputs and inducing inputs
        forward
        """

        W = find_weights(self.input_dim, self.output_dim, X)

        self.Z.assign(Z)

        Z_running = self.Z.value.copy().dot(W)
        X_running = X.copy().dot(W)

        if isinstance(self.mean_function, Linear):
            self.mean_function.A =W

        return Z_running, X_running

class OutputLayer(Layer):
    ###Maybe add __init__ with Y to do assertion on output_dim
    @defer_build()
    def initialize_forward(self, X, Z):
        """
        Initialize Layer and Propagate values of inputs and inducing inputs
        forward
        """

        self.Z.assign(Z)
