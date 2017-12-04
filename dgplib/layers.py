from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from gpflow import settings

from gpflow.decors import params_as_tensors, autoflow
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Linear, Zero
from gpflow.params import Parameter, Parameterized

class Layer(Parameterized):
    """
    The basic layer class. Handles input_dim and output_dim.
    """

    def __init__(self, input_dim, output_dim, num_inducing, kernel, name=None):
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
        self.Z = None

        self.kernel = kernel
        self.mean_function = None

        shape = (self.num_inducing, self.output_dim)

        self.q_mu = Parameter(np.zeros(shape))

        q_sqrt = np.dstack([np.eye(self.num_inducing)
                                 for _ in range(self.output_dim)])
        self.q_sqrt = Parameter(q_sqrt)

    @params_as_tensors
    def build_prior_KL(self, K):
        gauss_kl(self.q_mu, self.q_sqrt, K=K)

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        pass

def find_weights(input_dim, output_dim, X):
    """
    Find the weights of the Linear mean function based on input and output
    dimensions of the layer
    """

    if input_dim == output_dim:
        W = np.eye(input_dim)
    elif input_dim > output_dim:
        _, _, V = np.linalg.svd(X, full_matrices=False)
        W = V[:input_dim, :].T
    elif input_dim < output_dim:
        I = np.eye(input_dim)
        zeros = np.zeros((input_dim, output_dim - input_dim))
        W = np.concatenate([I, zeros], 1)

    return W

class InputLayer(Layer):
    def __init__(self, input_dim, output_dim, Z, num_inducing, kernel, name=None):
        """
        input_dim is an integer
        output_dim is an integer
        Z is a matrix of inducing inputs
        num_inducing is the number of inducing inputs
        kernel is a kernel object (or list of kernel objects)
        """

        super(InputLayer, self).__init__(input_dim, output_dim, num_inducing,
                                         kernel, name)
        self.Z = Parameter(Z)

        assert self.Z[0] == self.num_inducing
        assert self.Z[1] == self.input_dim

    def initialize_forward(self, X):
        """
        Initialize Layer and Propagate initialization forwards
        """

        W = find_weights(self.input_dim, self.output_dim, X)

        Z_running = self.Z.value.copy().dot(W)
        X_running = X.copy().dot(W)

        self.mean_function = Linear(A=W)

        return Z_running, X_running


class HiddenLayer(Layer):
    def initialize_forward(self, X, Z):
        """
        Initialize Layer and Propagate initialization forwards
        """

        W = find_weights(self.input_dim, self.output_dim, X)

        self.Z = Parameter(Z)

        Z_running = self.Z.value.copy().dot(W)
        X_running = X.copy().dot(W)

        self.mean_function = Linear(A=W)

        return Z_running, X_running

class OutputLayer(Layer):
    ###Maybe add __init__ with Y to do assertion on outout_dim
    def initialize_forward(self, X, Z):
        """
        Initialize Layer and Propagate initialization forwards
        """

        self.Z = Parameter(Z)
        self.mean_function = Zero()
