from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from gpflow import settings

from gpflow.conditionals import conditional
from gpflow.decors import params_as_tensors, autoflow, defer_build
from gpflow.features import inducingpoint_wrapper
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Linear, Zero
from gpflow.params import Parameter, Parameterized, ParamList

from .layers import Layer
from .layers import InputMixin, HiddenMixin, OutputMixin

class MultikernelLayer(Layer):
    """
    Inherits from Layer class. Can handle outputs from different priors.
    """

    @defer_build()
    def __init__(self, input_dim, output_dim, num_inducing, kernel_list,
                 share_Z=False, mean_function=None, multitask=False, name=None):

        if output_dim%len(kernel_list) != 0:
            raise ValueError("Output dimension must be a multiple of the number of kernels")

        super(MultikernelLayer, self).__init__(input_dim=input_dim,
                                    output_dim=output_dim,
                                    num_inducing=num_inducing,
                                    kernel=kernel_list,
                                    mean_function=mean_function,
                                    multitask=multitask,
                                    name=name)

        self.num_kernels = len(kernel_list)
        self._shared_Z = share_Z
        self.offset = int(self.output_dim/self.num_kernels)

        if not self._shared_Z:
            del self.feature
            if multitask:
                Z = np.zeros((self.num_inducing, self.input_dim+1))
            else:
                Z = np.zeros((self.num_inducing, self.input_dim))

            self.feature = ParamList([inducingpoint_wrapper(None, Z.copy()) for _ in range(self.num_kernels)])


    @params_as_tensors
    def build_prior_KL(self, K):
        if K:
            KL = 0.
            for i, k in enumerate(K):
                KL += gauss_kl_white(self.q_mu[:,(i*self.offset):((i+1)*self.offset)],
                                     self.q_sqrt[(i*self.offset):((i+1)*self.offset),:,:],
                                     K=k
                                    )
            return KL
        else:
            return gauss_kl(self.q_mu, self.q_sqrt, K=K)

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, stochastic=True):
        def f_conditional(Xnew, full_cov=False):
            mean = []
            var = []
            if self._shared_Z:
                feats = [self.feature for _ in range(self.num_kernels)]
            else:
                feats = [feat for feat in self.feature]
            for i, (k, feat) in enumerate(zip(self.kernel, feats)):
                m, v = conditional(Xnew, feat, k, self.q_mu[:,(i*self.offset):((i+1)*self.offset)],
                                   q_sqrt=self.q_sqrt[(i*self.offset):((i+1)*self.offset),:,:,],
                                   full_cov=full_cov,
                                   white=True)
                mean.append(m)

                #temporary fix
                if full_cov:
                    var.append(tf.transpose(v))
                else:
                    var.append(v)

            mean = tf.concat(mean, axis=-1) #NxK
            var = tf.concat(var, axis=-1) #NxK or NxNxK

            return mean + self.mean_function(Xnew), var

        def multisample_conditional(Xnew, full_cov=False):
            if full_cov:
                f = lambda a: f_conditional(a, full_cov=full_cov)
                mean, var = tf.map_fn(f, Xnew, dtype=(settings.tf_float,
                                                  settings.tf_float))
                return tf.stack(mean), tf.stack(var)
            else:
                #S, N, D = shape_as_list(Xnew)
                s = tf.shape(Xnew)
                X_flat = tf.reshape(Xnew, [s[0]*s[1], s[2]])
                mean, var = f_conditional(X_flat)
                return [tf.reshape(m, [s[0], s[1], -1]) for m in [mean, var]]

        if stochastic:
            mean, var = multisample_conditional(Xnew, full_cov)
        else:
            mean, var = f_conditional(Xnew, full_cov)

        return mean, var

class MultikernelInputLayer(MultikernelLayer, InputMixin):
    @defer_build()
    def initialize_forward(self, X, Z, multitask=False):
        """
        Initialize Layer and Propagate values of inputs and inducing inputs
        forward
        """
        if self._shared_Z:
            self.feature.Z.assign(Z)
        else:
            for feat in self.feature:
                feat.Z.assign(Z)

        X_running, Z_running, W = self.compute_inputs(X, Z, multitask)

        if isinstance(self.mean_function, Linear):
            self.mean_function.A = W
            self.mean_function.set_trainable(False)

        return X_running, Z_running


class MultikernelHiddenLayer(MultikernelLayer, HiddenMixin):
    @defer_build()
    def initialize_forward(self, X, Z, multitask=False):
        """
        Initialize Layer and Propagate values of inputs and inducing inputs
        forward
        """
        if self._shared_Z:
            self.feature.Z.assign(Z)
        else:
            for feat in self.feature:
                feat.Z.assign(Z)

        X_running, Z_running, W = self.compute_inputs(X, Z, multitask)

        if isinstance(self.mean_function, Linear):
            self.mean_function.A =W
            self.mean_function.set_trainable(False)

        return X_running, Z_running

class MultikernelOutputLayer(MultikernelLayer, OutputMixin):
    @defer_build()
    def initialize_forward(self, X, Z, multitask=False):
        """
        Initialize Layer and Propagate values of inputs and inducing inputs
        forward
        """

        if self._shared_Z:
            self.feature.Z.assign(Z)
        else:
            for feat in self.feature:
                feat.Z.assign(Z)

        return (None, None)
