import unittest

import numpy as np

from dgplib.layers import InputLayer, HiddenLayer, OutputLayer
from dgplib.models import Sequential

from gpflow.decors import defer_build
from gpflow.kernels import RBF
from gpflow.params import ParamList

class SequentialTest(unittest.TestCase):
    def test_initialization_with_empty(self):
        seq = Sequential()
        self.assertIsInstance(seq.layers, ParamList)

    @defer_build()
    def test_initialization_with_list(self):
        input_layer = InputLayer(2, 2, 10, RBF(2))
        hidden_layer_1 = HiddenLayer(2, 2, 10, RBF(2))
        hidden_layer_2 = HiddenLayer(2, 2, 10, RBF(2))
        output_layer = OutputLayer(2, 1, 10, RBF(2))

        with self.subTest():
            layer_list = [input_layer, hidden_layer_1, hidden_layer_2,
                          output_layer]
            try:
                seq = Sequential(layer_list)
            except Exception as e:
                print(e)
                self.fail("Initialisation with list of layers fails")

        # Test initilisation with incorrect layer structure
        with self.subTest():
            layer_list = [hidden_layer_1, hidden_layer_2, output_layer]
            with self.assertRaises(AssertionError):
                seq = Sequential(layer_list)

        # Test initilisation with incorrect layer structure
        # with self.subTest():
            # layer_list = [input_layer, hidden_layer_1, hidden_layer_2]
            # with self.assertRaises(AssertionError):
                # seq = Sequential(layer_list)

    @defer_build()
    def test_add_to_empty(self):
        input_layer = InputLayer(2, 2, 10, RBF(2))
        hidden_layer_1 = HiddenLayer(2, 2, 10, RBF(2))
        output_layer = OutputLayer(2, 1, 10, RBF(2))

        # Add input layer only
        with self.subTest():
            seq = Sequential()
            seq.add(input_layer)
            self.assertIs(seq.layers[-1], input_layer)

        # Add input layer and hidden layer
        with self.subTest():
            seq = Sequential()
            seq.add(input_layer)
            seq.add(hidden_layer_1)
            self.assertIs(seq.layers[0], input_layer)
            self.assertIs(seq.layers[1], hidden_layer_1)

        # Add input layer, hidden layer and output layer
        with self.subTest():
            seq = Sequential()
            seq.add(input_layer)
            seq.add(hidden_layer_1)
            seq.add(output_layer)
            self.assertIs(seq.layers[0], input_layer)
            self.assertIs(seq.layers[1], hidden_layer_1)
            self.assertIs(seq.layers[2], output_layer)

        # Add hidden layer as first layer
        with self.subTest():
            seq = Sequential()
            with self.assertRaises(AssertionError):
                seq.add(hidden_layer_1)

        # Add output layer as first layer
        with self.subTest():
            seq = Sequential()
            with self.assertRaises(AssertionError):
                seq.add(output_layer)

    @defer_build()
    def test_add_to_full(self):
        input_layer = InputLayer(2, 2, 10, RBF(2))
        hidden_layer_1 = HiddenLayer(2, 2, 10, RBF(2))
        hidden_layer_2 = HiddenLayer(2, 2, 10, RBF(2))
        hidden_layer_3 = HiddenLayer(3, 2, 10, RBF(3))
        output_layer = OutputLayer(2, 2, 10, RBF(2))


        # Add hidden layer with correct dimensions
        with self.subTest():
            layer_list = [input_layer, hidden_layer_1]
            seq = Sequential(layer_list)
            seq.add(hidden_layer_2)
            self.assertIs(seq.layers[-1], hidden_layer_2)

        # Add hidden layer with incorrect dimensions
        with self.subTest():
            layer_list = [input_layer, hidden_layer_1]
            seq = Sequential(layer_list)
            with self.assertRaises(AssertionError):
                seq.add(hidden_layer_3)

        # Add output layer with correct dimensions
        with self.subTest():
            layer_list = [input_layer, hidden_layer_1]
            seq = Sequential(layer_list)
            seq.add(output_layer)
            self.assertIs(seq.layers[-1], output_layer)

        # Add hidden layer after output layer
        with self.subTest():
            layer_list = [input_layer, output_layer]
            seq = Sequential(layer_list)
            with self.assertRaises(ValueError):
                seq.add(hidden_layer_1)

    @defer_build()
    def test_dims(self):
        input_layer = InputLayer(2, 3, 10, RBF(2))
        hidden_layer_1 = HiddenLayer(3, 2, 10, RBF(2))
        hidden_layer_2 = HiddenLayer(2, 1, 10, RBF(2))
        hidden_layer_3 = HiddenLayer(1, 2, 10, RBF(3))
        output_layer = OutputLayer(2, 1, 10, RBF(2))

        layer_list = [input_layer, hidden_layer_1, hidden_layer_2,
                      hidden_layer_3, output_layer]
        seq = Sequential(layer_list)
        dims = seq.get_dims()
        reference = [(2,3), (3,2), (2,1), (1,2), (2,1)]
        self.assertEqual(dims, reference)

    @defer_build()
    def test_initialize_params(self):
        input_layer = InputLayer(2, 2, 10, RBF(2))
        hidden_layer_1 = HiddenLayer(2, 2, 10, RBF(2))
        output_layer = OutputLayer(2, 1, 10, RBF(2))

        Z = np.ones((10, 2))
        X = np.ones((100, 2))

        seq = Sequential([input_layer, hidden_layer_1, output_layer])
        seq.initialize_params(X, Z)

        self.assertTrue(np.allclose(Z, seq.layers[0].Z.value))
