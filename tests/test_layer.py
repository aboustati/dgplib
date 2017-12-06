#temporary
#import sys
#sys.path.append('../dgplib/')
#import layers

import unittest

import numpy as np

from dgplib.layers import find_weights
from gpflow.kernels import White
from gpflow.test_util import GPflowTestCase

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

if __name__=='__main__':
    unittest.main()
