import unittest

import numpy as np

from dgplib.dgplib  import layers

from gpflow.kernels import White

class LayerTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_constructor(self):
        pass

    def test_build_prior_KL(self):
        K = White(1, 1)
        layer = layers.Layer(1, 1, 5, K)
        layer.Z = np.random.randn(5,1)
        #Run prior KL and expect answer to be 0

    def test_build_predict(self):
        #Run on small toy dataset and expect answer to be similar to SVGP
        pass
