from .layers import Layer
from .multiprocess_layers import MultiprocessLayer, ConcatinativeMultiprocessLayerMixin, AdditiveMultiprocessLayerMixin


class WeightedLayer(Layer):
    """
    The basic SVGP layer class with weights on the KL divergence.
    """

    def __init__(self, input_dim, output_dim, kernel, feature=None, weight=1.0,
                 num_inducing=None, share_Z=False, fixed_linear_mean_function=False,
                 mean_function=None, whiten=True, q_diag=False, q_mu=None, q_sqrt=None, name=None):

        super().__init__(input_dim=input_dim, output_dim=output_dim, kernel=kernel, feature=feature,
                         num_inducing=num_inducing, share_Z=share_Z,
                         fixed_linear_mean_function=fixed_linear_mean_function, mean_function=mean_function,
                         whiten=whiten, q_diag=q_diag, q_mu=q_mu, q_sqrt=q_sqrt, name=name)

        self.weight = weight

    def prior_kl(self):
        return self.weight * super().prior_kl()


class WeightedMultiprocessLayer(MultiprocessLayer):
    """
    Inherits from Layer class. Can handle outputs from different priors.
    """

    def __init__(self, input_dim, sublayer_output_dim, kernels, weights, feature=None,
                 num_inducing=None, share_Z=False, fixed_linear_mean_function=False,
                 mean_functions=None, whiten=True, q_diag=False, q_mu=None, q_sqrt=None, name=None):
        super(MultiprocessLayer, self).__init__(name=name)

        self.input_dim = input_dim
        self.sublayer_output_dim = sublayer_output_dim
        self.num_sublayers = len(kernels)

        if mean_functions is None:
            mean_functions = [None] * self.num_sublayers

        self.fixed_linear_mean_function = fixed_linear_mean_function

        sublayers = []
        for i in range(self.num_sublayers):
            sublayer = WeightedLayer(input_dim=self.input_dim, output_dim=self.sublayer_output_dim, kernel=kernels[i],
                             feature=feature, weight=weights[i], num_inducing=num_inducing, share_Z=share_Z,
                             fixed_linear_mean_function=self.fixed_linear_mean_function,
                             mean_function=mean_functions[i], whiten=whiten, q_diag=q_diag, q_mu=q_mu, q_sqrt=q_sqrt)
            sublayers.append(sublayer)

        self.sublayers = sublayers


class WeightedConcatinativeMultiprocessLayer(ConcatinativeMultiprocessLayerMixin, WeightedMultiprocessLayer):
    pass


class WeightedAdditiveMultiprocessLayer(AdditiveMultiprocessLayerMixin, WeightedMultiprocessLayer):
    pass

