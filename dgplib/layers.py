from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from gpflow import settings

from gpflow.params import Parameter, Parameterized
from gpflow.decors import params_as_tensors, autoflow

class Layer(Parameterized):
    """
    The basic layer class. Handles input_dim and output_dim.
    """

    def __init__(self, input_dim, output_dim, Z, kernel, mean_function, name=None):
        """
        input_dim is an integer
        output_dim is an integer
        Z is a matrix of inducing inputs
        kernel is a kernel object (or list of kernel objects)
        mean_fucntion is a mean_function object
        """

        super(Layer, self).__init__(name=name)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = Z.shape[0]
        self.Z = Parameter(Z)

        assert input_dim == Z.shape[1]

        self.kernel = kernel
        self.mean_function = mean_function

        shape = (self.num_inducing, self.output_dim)

        self.q_mu = Parameter(np.zeros(shape))

        q_sqrt = np.dstack([np.eye(self.num_inducing)
                                 for _ in range(self.output_dim)])
        self.q_sqrt = Parameter(q_sqrt)

    @params_as_tensors
    def build_prior_KL(self):
        pass

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        pass
