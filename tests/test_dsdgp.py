import pytest

import numpy as np

from dgplib.layers import Layer
from dgplib.cascade import Sequential

from dgplib import DSDGP

import gpflow
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian


rng = np.random.RandomState(42)


class Datum:
    num_data = 100
    num_inducing = 10
    num_samples = 4
    num_tasks = 2
    X_dim, Y_dim = 2, 1
    inner_dim = 3
    X = rng.randn(num_data, X_dim)
    Y = rng.randn(num_data, Y_dim)
    Z = rng.randn(num_inducing, X_dim)

    X_idx = rng.randint(0, num_tasks, (num_data, 1))
    Z_idx = rng.randint(0, num_tasks, (num_inducing, 1))

    multi_X = np.hstack([X, X_idx])
    multi_Z = np.hstack([Z, Z_idx])
    multi_Y = np.hstack([Y, X_idx])


def create_layer_utility(input_dim, output_dim):
    """
    Utility function to create layer object.
    """
    kernel = gpflow.kernels.mo_kernels.SharedIndependentMok(
        RBF(variance=1.0, lengthscale=1.0, active_dims=range(Datum.X_dim)), output_dimensionality=output_dim
    )
    layer = Layer(input_dim=input_dim, output_dim=output_dim, kernel=kernel, num_inducing=Datum.num_inducing)
    return layer


@pytest.fixture
def model():
    likelihood = Gaussian()
    input_layer = create_layer_utility(Datum.X_dim, Datum.inner_dim)
    output_layer = create_layer_utility(Datum.inner_dim, Datum.Y_dim)

    seq = Sequential([input_layer, output_layer])

    model = DSDGP(Z=Datum.Z, layers=seq, likelihood=likelihood)
    return model


@pytest.fixture
def multitask_model():
    likelihood = Gaussian()
    input_layer = create_layer_utility(Datum.X_dim, Datum.inner_dim)
    output_layer = create_layer_utility(Datum.inner_dim, Datum.Y_dim)

    seq = Sequential([input_layer, output_layer])

    model = DSDGP(Z=Datum.multi_Z, layers=seq, likelihood=likelihood, multitask=True)
    return model


@pytest.mark.parametrize('full_cov', [True, False])
def test_predict_all_layers(model, full_cov):
    """
    Test the predict all layers method which is called by all other predict methods.
    """
    with pytest.raises(ValueError):
        model.predict_all_layers(Xnew=Datum.X, num_samples=Datum.num_samples, full_cov=full_cov)

    model.initialize_layers_from_data(Datum.X)

    fs, fmeans, fvars = model.predict_all_layers(Xnew=Datum.X, num_samples=Datum.num_samples, full_cov=full_cov)
    dims = [Datum.inner_dim, Datum.Y_dim]
    for f, m, v, i in zip(fs, fmeans, fvars, dims):
        assert m.shape == f.shape
        if full_cov:
            assert v.shape == (Datum.num_samples, i, Datum.num_data, Datum.num_data)
        else:
            assert v.shape == m.shape
        assert m.shape == (Datum.num_samples, Datum.num_data, i)
        np.testing.assert_array_less(np.full_like(v, -1e-6), v)


@pytest.mark.parametrize('full_cov', [True, False])
def test_predict_all_layers_multitask(multitask_model, full_cov):
    """
    Test the predict all layers method for multitask model which is called by all other predict methods.
    """
    with pytest.raises(ValueError):
        multitask_model.predict_all_layers(Xnew=Datum.multi_X, num_samples=Datum.num_samples, full_cov=full_cov)

    multitask_model.initialize_layers_from_data(Datum.multi_X)

    fs, fmeans, fvars = multitask_model.predict_all_layers(Xnew=Datum.multi_X, num_samples=Datum.num_samples, full_cov=full_cov)
    dims = [Datum.inner_dim, Datum.Y_dim]
    for f, m, v, i in zip(fs, fmeans, fvars, dims):
        print(m.shape)
        assert f.shape == (Datum.num_samples, Datum.num_data, i + 1)
        if full_cov:
            assert v.shape == (Datum.num_samples, i, Datum.num_data, Datum.num_data)
        else:
            assert v.shape == m.shape
        assert m.shape == (Datum.num_samples, Datum.num_data, i)
        np.testing.assert_array_less(np.full_like(v, -1e-6), v)


def test_log_likelihood(model):
    """
    Tests the log_likelihood method.
    """
    model.initialize_layers_from_data(Datum.X)

    likelihood_value = model.log_likelihood(Datum.X, Datum.Y, Datum.num_samples)

    assert np.isscalar(likelihood_value.numpy())

