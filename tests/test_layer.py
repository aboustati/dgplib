import unittest

import numpy as np

from dgplib.layers import find_weights, InputLayer, HiddenLayer, OutputLayer

from gpflow.decors import defer_build
from gpflow.kernels import White, RBF
from gpflow.mean_functions import Linear

class LayerTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_build_predict(self):
        #Run on small toy dataset and expect answer to be similar to SVGP
        pass

class WeightsTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[2,4], [1,3], [0,0], [0,0]])

    def test_input_equals_output(self):
        W = find_weights(2, 2, self.X)
        with self.subTest():
            self.assertEqual(W.shape, (2,2))
        with self.subTest():
            self.assertTrue(np.allclose(W, np.eye(2)))

    def test_input_greater_output(self):
        W = find_weights(2, 1, self.X)
        V = np.array([-0.4, -0.91]).reshape((2,1))
        with self.subTest():
            self.assertEqual(W.shape, (2,1))
        with self.subTest():
            self.assertTrue(np.allclose(W, V, atol=1e-2))

    def test_input_less_output(self):
        W = find_weights(1, 2, self.X)
        V = np.array([1, 0])[None, :]
        with self.subTest():
            self.assertEqual(W.shape, (1,2))
        with self.subTest():
            self.assertTrue(np.allclose(W, V))

class InputLayerTest(unittest.TestCase):
    @defer_build()
    def setUp(self):
        self.rng = np.random.RandomState(42)
        kernel = RBF(2)
        input_dim = 2
        output_dim = 2
        self.W0 = np.zeros((input_dim, output_dim))
        mean_function = Linear(A=self.W0)
        self.Z = self.rng.randn(5,2)
        num_inducing = 5

        self.layer = InputLayer(input_dim=input_dim,
                                output_dim=output_dim,
                                num_inducing=num_inducing,
                                kernel=kernel,
                                mean_function=mean_function)

        self.X = self.rng.randn(10,2)

    def test_initialize_forward(self):
        X_running, Z_running = self.layer.initialize_forward(self.X, self.Z)

        with self.subTest():
            self.assertFalse(np.allclose(self.layer.mean_function.A.value, self.W0))

        with self.subTest():
           self.assertTrue(np.allclose(Z_running, self.Z))

        with self.subTest():
           self.assertTrue(np.allclose(self.layer.feature.Z.value, self.Z))

        with self.subTest():
            self.assertTrue(np.allclose(X_running, self.X))

class HiddenLayerTest(unittest.TestCase):
    @defer_build()
    def setUp(self):
        self.rng = np.random.RandomState(42)
        kernel = RBF(2)
        input_dim = 2
        output_dim = 2
        self.W0 = np.zeros((input_dim, output_dim))
        mean_function = Linear(A=self.W0)
        self.Z = self.rng.randn(5, 2)
        num_inducing = 5

        self.layer = HiddenLayer(input_dim=input_dim,
                                 output_dim=output_dim,
                                 num_inducing=num_inducing,
                                 kernel=kernel,
                                 mean_function=mean_function)

        self.X = self.rng.randn(10, 2)

    def test_initialize_forward(self):
        X_running, Z_running = self.layer.initialize_forward(self.X, self.Z)

        with self.subTest():
            self.assertFalse(np.allclose(self.layer.mean_function.A.value, self.W0))

        with self.subTest():
           self.assertTrue(np.allclose(Z_running, self.Z))

        with self.subTest():
            self.assertTrue(np.allclose(X_running, self.X))

        with self.subTest():
            self.assertTrue(np.allclose(self.layer.feature.Z.value, self.Z))

class OutputLayerTest(unittest.TestCase):
    @defer_build()
    def setUp(self):
        self.rng = np.random.RandomState(42)
        kernel = RBF(2)
        input_dim = 2
        output_dim = 2
        mean_function = None
        self.Z = self.rng.randn(5, 2)
        num_inducing = 5

        self.layer = OutputLayer(input_dim=input_dim,
                                 output_dim=output_dim,
                                 num_inducing=num_inducing,
                                 kernel=kernel,
                                 mean_function=mean_function)

        self.X = self.rng.randn(10,2)

    def test_initialize_forward(self):
        _ = self.layer.initialize_forward(self.X, self.Z)

        with self.subTest():
           self.assertTrue(np.allclose(self.layer.feature.Z.value, self.Z))

if __name__=='__main__':
    unittest.main()
