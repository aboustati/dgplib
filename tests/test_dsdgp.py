import unittest

import numpy as np
import gpflow

from dgplib.layers import InputLayer, OutputLayer
from dgplib.cascade import Sequential

from dgplib import DSDGP

from gpflow.decors import defer_build, name_scope
from gpflow.kernels import RBF, White
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Linear

class TestDSDGP(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)

        self.Ns = 300
        #self.Xs = np.linspace(-0.5, 1.5, Ns)[:, None]

        self.N, self.M = 50, 25
        self.X = self.rng.uniform(0, 1, self.N)[:, None]
        self.Z = self.rng.uniform(0, 1, self.M)[:, None]
        f_step = lambda x: 0. if x<0.5 else 1.

        self.Y = np.reshape([f_step(x) for x in self.X], self.X.shape) \
                 + self.rng.randn(*self.X.shape)*1e-2

    def test_contructor(self):
        input_layer = InputLayer(input_dim=1, output_dim=1,
                                 num_inducing=self.M, kernel=RBF(1)+White(1))
        output_layer = OutputLayer(input_dim=1, output_dim=1,
                                   num_inducing=self.M, kernel=RBF(1)+White(1))

        seq = Sequential([input_layer, output_layer])

        try:
            model = DSDGP(X=self.X, Y=self.Y, Z=self.Z, layers=seq, likelihood=Gaussian())
        except Exception as e:
            print(e)
            self.fail('DSDGP contructor fails')

    @name_scope('dsdgp_optimizer')
    def test_optimize(self):
        with defer_build():
            input_layer = InputLayer(input_dim=1, output_dim=1,
                                     num_inducing=self.M, kernel=RBF(1)+White(1))
            output_layer = OutputLayer(input_dim=1, output_dim=1,
                                       num_inducing=self.M, kernel=RBF(1)+White(1))

            seq = Sequential([input_layer, output_layer])

            model = DSDGP(X=self.X, Y=self.Y, Z=self.Z, layers=seq, likelihood=Gaussian())
        model.compile()
        before = model.compute_log_likelihood()
        opt = gpflow.train.AdamOptimizer(0.01)
        opt.minimize(model, maxiter=100)
        after = model.compute_log_likelihood()
        self.assertGreaterEqual(after, before)

class TestMethods(unittest.TestCase):
    def prepare(self):
        N = 100
        M = 10
        rng = np.random.RandomState(42)
        X = rng.randn(N, 2)
        Y = rng.randn(N, 1)
        Z = rng.randn(M, 2)
        Xs = rng.randn(M, 2)
        lik = Gaussian()
        input_layer = InputLayer(input_dim=2, output_dim=1,
                                 num_inducing=M, kernel=RBF(2)+White(2),
                                 mean_function=Linear(A=np.ones((2,1))))
        output_layer = OutputLayer(input_dim=1, output_dim=1,
                                   num_inducing=M, kernel=RBF(1)+White(1))

        seq = Sequential([input_layer, output_layer])

        model = DSDGP(X=X, Y=Y, Z=Z, layers=seq, likelihood=lik)
        model.compile()
        return model, Xs

    def test_build(self):
        model, _ = self.prepare()
        self.assertEqual(model.is_built_coherence(), gpflow.Build.YES)

    def test_predict_f(self):
        model, Xs = self.prepare()
        mu, sigma = model.predict_f(Xs, 1)
        with self.subTest():
            self.assertEqual(mu.shape, sigma.shape)
        with self.subTest():
            self.assertEqual(mu.shape, (1, 10, 1))
        with self.subTest():
            np.testing.assert_array_less(np.full_like(sigma, -1e-6), sigma)

    def test_predict_f_full_cov(self):
        model, Xs = self.prepare()
        mu, sigma = model.predict_f_full_cov(Xs, 1)
        with self.subTest():
            self.assertEqual(mu.shape, (1, 10, 1))
        with self.subTest():
            self.assertEqual(sigma.shape, (1, 10, 10, 1))
        with self.subTest():
            np.testing.assert_array_less(np.full_like(sigma, -1e-6), sigma)

    def test_predict_all_layers(self):
        model, Xs = self.prepare()
        fs, fmeans, fvars = model.predict_all_layers(Xs, 1)
        dims = [1, 1]
        for f, m, v, i in zip(fs, fmeans, fvars, dims):
            with self.subTest():
                self.assertEqual(m.shape, f.shape)
            with self.subTest():
                self.assertEqual(m.shape, v.shape)
            with self.subTest():
                self.assertEqual(m.shape, (1, 10, i))
            with self.subTest():
                np.testing.assert_array_less(np.full_like(v, -1e-6), v)

    def test_predict_all_layers_full_cov(self):
        model, Xs = self.prepare()
        fs, fmeans, fvars = model.predict_all_layers_full_cov(Xs, 1)
        dims = [1, 1]
        for f, m, v, i in zip(fs, fmeans, fvars, dims):
            with self.subTest():
                self.assertEqual(f.shape, (1, 10, i))
            with self.subTest():
                self.assertEqual(m.shape, (1, 10, i))
            with self.subTest():
                self.assertEqual(v.shape, (1, 10, 10, 1))
            with self.subTest():
                np.testing.assert_array_less(np.full_like(v, -1e-6), v)

    def test_predict_f_samples(self):
        model, Xs = self.prepare()
        fs = model.predict_f_samples(Xs, 10)
        with self.subTest():
            self.assertEqual(fs.shape, (10, 10, 1))

    def test_predict_y(self):
        model, Xs = self.prepare()
        mu, sigma = model.predict_y(Xs)
        with self.subTest():
            self.assertEqual(mu.shape, sigma.shape)
        with self.subTest():
            self.assertEqual(mu.shape, (1, 10, 1))
        with self.subTest():
            np.testing.assert_array_less(np.full_like(sigma, -1e-6), sigma)
