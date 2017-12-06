#temporary
#import sys
#sys.path.append('../dgplib/')
#import layers

import unittest

import numpy as np

from dgplib import layers
from gpflow.kernels import White

class LayerTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_constructor(self):
        pass

    def test_build_prior_KL(self):
        K = White(1, variance=1.0)
        layer = layers.Layer(input_dim=1, output_dim=1, num_inducing=5, kernel=K)
        layer.build_prior_KL(None)
        #Run prior KL and expect answer to be 0

    def test_build_predict(self):
        #Run on small toy dataset and expect answer to be similar to SVGP
        pass


if __name__=='__main__':
    unittest.main()
