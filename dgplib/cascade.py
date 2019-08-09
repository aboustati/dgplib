from gpflow.base import Module
from .layers import Layer
from .multiprocess_layers import MultiprocessLayer


class Sequential(Module):
    """
    Linear Stack of layers
    """

    def __init__(self, layers=None, name=None):
        """
        :param layers: list of Layer objects
        """

        super().__init__(name=name)

        self._initialized = False
        self.constituents = []

        if layers:
            for layer in layers:
                self.add(layer)

    @property
    def initialized(self):
        return self._initialized

    @initialized.setter
    def initialized(self, value):
        if not self.initialized:
            if value:
                raise ValueError("Cannot overwrite initialization for uninitialized models")
        self._initialized = value

    def _valid_input(self, layer):
        """
        Checks if input to the cascade object is valid
        """
        assert isinstance(layer, (Layer, MultiprocessLayer))

        if self.initialized:
            raise ValueError('Cannot add more layers to initialized model')

        if self.constituents:  # if list is not empty
            assert self.constituents[-1].output_dim == layer.input_dim, """Input
            dimensions of layer must be equal to the output dimensions of the
            preceding layer"""

    def add(self, layer):
        """
        Adds a layer instance on top of the layer stack.

        :param layer: Layer object
        """
        self._valid_input(layer)
        self.constituents.append(layer)

    def get_dims(self):
        """
        Get a list of the dimensions of the constituent layers.
        """
        dims = [(l.input_dim, l.output_dim) for l in self.constituents]

        return dims

    def initialize_params(self, X, Z):
        """
        Handles the initialization of inducing inputs in the Sequential cascade
        """
        Z_current = Z.copy()
        X_next, Z_next, W = self.constituents[0].propagate_inputs_and_features(X, Z)
        self.constituents[0].initialize_features(Z_current)
        if self.constituents[0].fixed_linear_mean_function:
            self.constituents[0].initialize_linear_mean_function_weights(W)
        for layer in self.constituents[1:]:
            Z_current = Z_next
            X_next, Z_next, W = layer.propagate_inputs_and_features(X_next, Z_current)
            layer.initialize_features(Z_current)
            if layer.fixed_linear_mean_function:
                layer.initialize_linear_mean_function_weights(W)

        print('Model Parameters Initialized')
        self._initialized = True


