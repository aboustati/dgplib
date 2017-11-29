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

    def __init__(self, input_dim, output_dim, Z, name=None):
        """
        input_dim is an integer
        output_dim is an integer
        Z is a matrix of Inducing inputs
        """

        super(Layer, self).__init__(name=name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Z = Parameter(Z)

    @params_as_tensors
    def build_prior_KL(self):
        pass

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        pass
