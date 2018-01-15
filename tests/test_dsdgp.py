import unittest

import numpy as np

from dgplib.layers import InputLayer, OutputLayer, HiddenLayer
from dgplib.models import Sequential

from dgplib import DSDGP

from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Linear

class TestDSDGP(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.seed(42)

        self.Ns = 300
        #self.Xs = np.linspace(-0.5, 1.5, Ns)[:, None]

        self.N, self.M = 50, 25
        self.X = np.random.uniform(0, 1, self.N)[:, None]
        self.Z = np.random.uniform(0, 1, self.M)[:, None]
        f_step = lambda x: 0. if x<0.5 else 1.

        self.Y = np.reshape([f_step(x) for x in self.X], self.X.shape) \
                 + np.random.randn(*self.X.shape)*1e-2

    def test_contructor(self):
        input_layer = InputLayer(input_dim=1, output_dim=1, Z=self.Z,
                                 num_inducing=self.M, kernel=RBF(1))
        output_layer = OutputLayer(input_dim=1, output_dim=1,
                                   num_inducing=self.M, kernel=RBF(1))

        seq = Sequential([input_layer, output_layer])

        try:
            model = DSDGP(X=self.X, Y=self.Y, Z=self.Z, layers=seq, likelihood=Gaussian())
        except Exception as e:
            print(e)
            self.fail('DSDGP contructor fails')
