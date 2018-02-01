import numpy as np
import tensorflow as tf

from gpflow.params import Parameterized, ParamList
from .layers import Layer, InputLayer, OutputLayer, HiddenLayer
from .multikernel_layers import MultikernelLayer, MultikernelInputLayer, MultikernelOutputLayer, MultikernelHiddenLayer

class Sequential(Parameterized):
    """
    Linear Stack of layers
    """

    def __init__(self, layers=None, name=None):
        """
        - layers is a list of layer objects.
        """

        super(Sequential, self).__init__(name=name)

        self._initialized = False #Change into property later
        self.layers = ParamList([])

        if layers:
            for layer in layers:
                self.add(layer)

    def _valid_input(self, layer):
        assert isinstance(layer, Layer)

        if self._initialized:
            raise ValueError('Cannot add more layers to initialized model')

        if not self.layers:
            #Temporary Hack
            assert isinstance(layer, InputLayer) or isinstance(layer,
                                                               MultikernelInputLayer), """First layer must be an
            Input Layer"""
        else:
            #Temporary Hack
            if isinstance(self.layers[-1], OutputLayer) or \
            isinstance(self.layers[-1], MultikernelOutputLayer):
                raise ValueError('Cannot add layers after an Output Layer')

            assert self.layers[-1].output_dim == layer.input_dim, """Input
            dimensions of layer must be equal to the output dimensions of the
            preceding layer"""

    def add(self, layer):
        """
        Adds a layer instance on top of the layer stack.

        - layer is an instance of an object that inherits from Layer
        """
        self._valid_input(layer)
        self.layers.append(layer)

    def get_dims(self):
        """
        Get a list of the dimensions of the constituent layers.
        """
        dims = [(l.input_dim, l.output_dim) for l in self.layers]

        return dims

    def initialize_params(self, X, Z):
        X_running, Z_running = self.layers[0].initialize_forward(X, Z)
        for layer in self.layers[1:]:
            X_running, Z_running = layer.initialize_forward(X_running, Z_running)

        print('Model Parameters Initialized')
        self._initialized = True
