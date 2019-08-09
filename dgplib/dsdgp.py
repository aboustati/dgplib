import tensorflow as tf

from gpflow.models import GPModel

from .layers import Layer
from .cascade import Sequential


class DSDGP(GPModel):
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
    def __init__(self, likelihood, Z, layers=None, kernels=None, dims=None,
                 num_latent=1, num_data=None, multitask=False):
        """
        :param likelihood: GPflow likelihood object
        :param Z: Initial value of inducing inputs
        :param layers: list of Layer objects
        :param kernels: list of kernel objects (cannot be used if layers are specified)
        :param dims: list of tuples for layer dims to be used when kernels are specified
        :param num_latent: latent dimension of final layer
        :param num_data: number of data
        :param multitask: True if multitask model is required (propagates task labels)
        """
        if kernels is not None:
            assert len(dims) == len(kernels) + 1
            assert layers is None, "Cannot initialise with kernels and layers simultaneously"
            layers = Sequential()
            for i, kernel in enumerate(kernels):
                fix_linear_mf = False if i == len(kernels) - 1 else True
                layer = Layer(
                    input_dim=dims[i],
                    output_dim=dims[i+1],
                    kernel=kernel,
                    share_Z=False,
                    fixed_linear_mean_function=fix_linear_mf
                )
                layers.add(layer)

        super().__init__(
            kernel=None,
            likelihood=likelihood,
            mean_function=None,
            num_latent=num_latent
        )

        # Kernels and mean functions are handled by layers
        del self.kernel
        del self.mean_function

        self.num_data = num_data

        self.initial_Z = Z
        self.layers = layers

        self.dims = self.layers.get_dims()
        self._multitask = multitask

    @property
    def initialized(self):
        return self.layers.initialized

    @property
    def multitask(self):
        return self._multitask

    def initialize_layers_from_data(self, X):
        self.layers.initialize_params(X, self.initial_Z)

    def _propagate(self, Xnew, full_cov=False, num_samples=1):
        """
        Propagate points Xnew through DGP cascade.
        """
        if not self.initialized:
            raise ValueError("Must initialize before calling this method")

        F_samples = [tf.stack([Xnew] * num_samples)]
        F_mus, F_vars = [], []
        for layer in self.layers.constituents:
            f_samples, f_mus, f_vars = [], [], []
            for s in range(num_samples):
                f_sample, f_mu, f_var = layer.predict_f_samples(F_samples[-1][s, ...], full_cov=full_cov, num_samples=1)

                if self.multitask:
                    f_sample = tf.concat([f_sample, Xnew[None, :, -1:]], axis=-1)

                f_samples.append(f_sample)
                f_mus.append(f_mu)
                f_vars.append(f_var)

            F_samples.append(tf.concat(f_samples, axis=0))
            F_mus.append(tf.stack(f_mus))
            F_vars.append(tf.stack(f_vars))

        return F_samples[1:], F_mus, F_vars

    def predict_all_layers(self, Xnew, full_cov=False, num_samples=1):
        """
        Predicts and produces posterior samples from all layers
        """
        Fs, Fmeans, Fvars = self._propagate(Xnew, full_cov, num_samples)
        return Fs, Fmeans, Fvars

    def predict_f(self, Xnew, full_cov=False, num_samples=1):
        """
        Returns the posterior for the final layer
        """
        Fs, Fmeans, Fvars = self.predict_all_layers(Xnew, full_cov, num_samples)
        return Fmeans[-1], Fvars[-1]

    def predict_f_samples(self, Xnew, full_cov=False, num_samples=1):
        """
        Produces posterior samples from the final layer
        """
        Fs, Fmeans, Fvars = self.predict_all_layers(Xnew, full_cov, num_samples)
        return Fs[-1]

    def prior_kl(self):
        """
        KL(q(u)||p(u))
        """
        return tf.reduce_sum([l.prior_kl() for l in self.layers.constituents])

    def log_likelihood(self, X: tf.Tensor, Y:tf. Tensor, num_samples: int = 1) -> tf.Tensor:
        """
        Evaluates bound on the log marginal likelihood
        """
        f_mean, f_var = self.predict_f(X, full_cov=False, num_samples=num_samples)  # SxNxD, SXNxD

        var_exp = [
            self.likelihood.variational_expectations(f_mean[s, :, :], f_var[s, :, :], Y) for s in range(num_samples)
        ]
        var_exp = tf.reduce_mean(tf.stack(var_exp), axis=0)
        assert var_exp.shape == (X.shape[0], self.num_latent)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, var_exp.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], var_exp.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, var_exp.dtype)

        var_exp = var_exp * scale

        return tf.reduce_sum(var_exp) - self.prior_kl()

    def elbo(self, X: tf.Tensor, Y: tf.Tensor, num_samples: int = 1) -> tf.Tensor:
        """
        This returns the evidence lower bound (ELBO) of the log marginal likelihood.
        """
        return -self.neg_log_marginal_likelihood(X, Y, num_samples)
