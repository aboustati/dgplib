import numpy as np
import tensorflow as tf

from gpflow.params import Parameterized, ParamList
from .layers import Layer, InputLayer, OutputLayer, HiddenLayer

class Sequential(Parameterized):
    """
    Linear Stack of layers
    """

    def __init__(self, layers=None, name=None):
        """
        layers is a list of layer objects
        """
        self.layers = ParamList([])

        if layers:
            assert isinstance(layers[0], InputLayer)
            assert isinstance(layers[-1], OutputLayer)
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        """
        Adds a layer instance on top of the layer stack

        layer is an instance of an object that inherits from Layer
        """
        assert issubclass(layer, Layer)

        if not self.layers:
            assert isinstance(layer, InputLayer)
        else:
            assert self.layers[-1].output_dim == layer.input_dim

        self.layers.append(layer)
