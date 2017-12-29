import numpy as np
import tensorflow as tf

from gpflow import settings

from gpflow.decors import defer_build
from gpflow.mean_functions import Zero
from gpflow.model import Model
from gpflow.params import DataHolder, Minibatch

from .utils import normal_sample, tile_over_samples

class DSDGP(Model):
    """
    Doubly Stochastic Deep Gaussian Process Model.

    Key reference:
        @incollection{NIPS2017_7045,
            title = {Doubly Stochastic Variational Inference for Deep Gaussian
            Processes},
            author = {Salimbeni, Hugh and Deisenroth, Marc},
            booktitle = {Advances in Neural Information Processing Systems 30},
            editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and
            R. Fergus and S. Vishwanathan and R. Garnett},
            pages = {4591--4602},
            year = {2017},
            publisher = {Curran Associates, Inc.},
            url =
            {http://papers.nips.cc/paper/7045-doubly-stochastic-variational-inference-for-deep-gaussian-processes.pdf}
        }
    """
    @defer_build()
    def __init__(self, X, Y, Z, layers, likelihood,
                 num_latent_Y=None,
                 minibatch_size=None,
                 num_samples=1,
                 mean_function=Zero(),
                 name=None):
        """
        - X is a data matrix, size N x D.
        - Y is a data matrix, size N x R.
        - Z is a matrix of inducing inputs, size M x D.
        - layers is an instance of Sequential containing the layer structure of
        the DGP.
        - likelihood is an instance of the gpflow likehood object.
        - num_latent_Y is the number of latent processes to use.
        - minibatch_size, if not None turns of minibatching with that size.
        - num_samples is the number of Monte Carlo samples to use.
        - mean_function is an instance of the gpflow mean_function object,
        corresponds to the mean function of the final layer.
        - name is the name of the TensforFlow object.
        """

        super(DSDGP, self).__init__(name=name)

        assert X.shape[0] == Y.shape[0]
        assert Z.shape[1] == X.shape[1]

        self.num_data, D_X = X.shape
        self.num_samples = num_samples
        self.D_Y = num_latent_Y or Y.shape[1]

        self.dims = layers.get_dims() #Not implemented yet

        self.likelihood = likelihood

        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)

        self.X, self.Y = X, Y
