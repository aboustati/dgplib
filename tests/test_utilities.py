import pytest
import numpy as np

from dgplib.utilities import find_linear_mf_weights


rng = np.random.RandomState(42)

N = 100
D = 5
X = rng.randn(N, D)


@pytest.fixture
def data():
    return X


@pytest.mark.parametrize('output_dim', [4, 5, 6])
def test_mf_weights_no_index_column(data, output_dim):
    """
    Tests mean function weight computation without an index column in the data
    """
    input_dim = D
    W = find_linear_mf_weights(input_dim=input_dim, output_dim=output_dim, X=data)
    assert W.shape == (input_dim, output_dim)


@pytest.mark.parametrize('output_dim', [3, 4, 5])
def test_mf_weights_with_index_column(data, output_dim):
    """
    Tests mean function weight computation with an index column in the data
    """
    input_dim = D - 1
    W = find_linear_mf_weights(input_dim=input_dim, output_dim=output_dim, X=data)
    assert W.shape == (input_dim + 1, output_dim)

