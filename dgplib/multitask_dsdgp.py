import numpy as np
import tensorflow as tf

from gpflow import settings

from gpflow.decors import autoflow, defer_build, params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.models import Model
from gpflow.params import DataHolder, Minibatch

from .dsdgp import DSDGP
from .utils import normal_sample, tile_over_samples

class MultitaskDSDGP(Model):
    @params_as_tensors
    def _propagate(self, Xnew, full_cov=False, num_samples=1):
        """
        Propagate points Xnew through DGP cascade.
        """
        Fs = [tile_over_samples(Xnew, num_samples), ]
        Fmeans, Fvars = [], []
        for layer in self.layers.layers:
            mean, var = layer._build_predict(Fs[-1], full_cov, stochastic=True)
            F = normal_sample(mean, var, full_cov=full_cov)
            #propagation of task labels
            F = tf.concat([F, Fs[-1][:,:,-1:]], axis=-1)

            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars

