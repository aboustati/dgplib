import pytest

import numpy as np

from dgplib.layers import Layer
from dgplib.cascade import Sequential

import gpflow
from gpflow.kernels import RBF


class Datum:
    num_inducing = 10
    num_data = 100
    dim = 2

    Z = np.ones((num_inducing, dim))
    X = np.ones((num_data, dim))


def create_layer_utility(input_dim, output_dim):
    """
    Utility function to create layer object.
    """
    kernel = gpflow.kernels.mo_kernels.SharedIndependentMok(
        RBF(variance=1.0, lengthscale=1.0), output_dimensionality=output_dim
    )
    layer = Layer(input_dim=input_dim, output_dim=output_dim, kernel=kernel, num_inducing=Datum.num_inducing)
    return layer


def test_add_to_empty():
    """
    Tests initializing the sequential cascade structure with an empty list of layers.
    """
    input_layer = create_layer_utility(2, 2)
    hidden_layer_1 = create_layer_utility(2, 2)
    output_layer = create_layer_utility(2, 1)

    # Add input layer only
    seq = Sequential()

    seq.add(input_layer)
    assert seq.constituents[-1] is input_layer

    seq.add(hidden_layer_1)
    assert seq.constituents[-1] is hidden_layer_1

    seq.add(output_layer)
    assert seq.constituents[-1] is output_layer


def test_add_to_full():
    """
    Tests adding additional layers to a a sequential cascade structure.
    """
    input_layer = create_layer_utility(2, 2)
    hidden_layer_1 = create_layer_utility(2, 2)
    hidden_layer_2 = create_layer_utility(2, 2)
    hidden_layer_3 = create_layer_utility(3, 2)

    # Add hidden layer with correct dimensions
    layer_list = [input_layer, hidden_layer_1]
    seq = Sequential(layer_list)
    seq.add(hidden_layer_2)
    assert seq.constituents[-1] is hidden_layer_2

    # Add hidden layer with incorrect dimensions
    layer_list = [input_layer, hidden_layer_1]
    seq = Sequential(layer_list)
    with pytest.raises(AssertionError):
        seq.add(hidden_layer_3)


def test_dims():
    """
    Tests the get_dim utility.
    """
    input_layer = create_layer_utility(2, 3)
    hidden_layer_1 = create_layer_utility(3, 2)
    hidden_layer_2 = create_layer_utility(2, 1)
    hidden_layer_3 = create_layer_utility(1, 2)
    output_layer = create_layer_utility(2, 1)

    layer_list = [input_layer, hidden_layer_1, hidden_layer_2,
                  hidden_layer_3, output_layer]
    seq = Sequential(layer_list)
    dims = seq.get_dims()
    reference = [(2, 3), (3, 2), (2, 1), (1, 2), (2, 1)]
    assert dims == reference


def test_initialize_params():
    """
    Tests the Layer initialization utility.
    """
    input_layer = create_layer_utility(2, 2)
    hidden_layer_1 = create_layer_utility(2, 2)
    output_layer = create_layer_utility(2, 1)

    seq = Sequential([input_layer, hidden_layer_1, output_layer])
    seq.initialize_params(Datum.X, Datum.Z)

    np.testing.assert_allclose(Datum.Z, seq.constituents[0].feature.features[0].Z.numpy())

