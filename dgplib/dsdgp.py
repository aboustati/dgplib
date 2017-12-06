import numpy as np
import tensorflow as tf

from gpflow import settings

from gpflow.mean_functions import Zero
from gpflow.model import Model
from gpflow.params import DataHolder, Minibatch

from .utils import normal_sample, tile_over_samples

class DSDGP(Model):
    def __init__(self, X, Y, Z, layers, likelihood,
                 num_latent_Y=None,
                 minibatch_size=None,
                 num_samples=1,
                 mean_function=Zero(),
                 name=None):

        super(DSDGP, self).__init__(name=name)

        assert X.shape[0] == Y.shape[0]
        assert Z.shape[1] == X.shape[1]

        self.num_data, D_X = X.shape
        self.num_samples = num_samples
        self.D_Y = num_latent_Y or Y.shape[1]

        #self.dims After sequential
