"""
Testing against GPflow's SVGP implementation as reference
"""

import pytest
import numpy as np
import tensorflow as tf

from dgplib.layers import Layer
from dgplib.utilities import find_linear_mf_weights

import gpflow
from gpflow.features import SharedIndependentMof, InducingPoints
from gpflow.kernels import RBF
from gpflow.mean_functions import Linear


rng = np.random.RandomState(42)


class Datum:
    input_dim = 5
    output_dim = 3
    num_inducing = 10
    num_data = 20
    noise_variance = 1.0
    lengthscale = 2.3
    signal_variance = 0.5
    q_mu = rng.randn(num_inducing, output_dim)
    q_sqrt = np.stack([np.tril(rng.rand(num_inducing, num_inducing))] * output_dim)

    X = rng.randn(num_data, input_dim)
    Z = rng.randn(num_inducing, input_dim)


@pytest.fixture
def reference_model():
    kernel = RBF(variance=Datum.signal_variance, lengthscale=Datum.lengthscale)
    likelihood = gpflow.likelihoods.Gaussian(variance=Datum.noise_variance)
    features = Datum.Z.copy()

    W = find_linear_mf_weights(input_dim=Datum.input_dim, output_dim=Datum.output_dim, X=Datum.X)

    model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood, feature=features, mean_function=Linear(A=W),
                               q_mu=Datum.q_mu, q_sqrt=Datum.q_sqrt)
    return model


def create_layer_utility(feature=None, share_Z=False, fixed_mf=True):
    kernel = gpflow.kernels.mo_kernels.SharedIndependentMok(
        RBF(variance=Datum.signal_variance, lengthscale=Datum.lengthscale), output_dimensionality=Datum.output_dim
    )
    layer = Layer(input_dim=Datum.input_dim, output_dim=Datum.output_dim, kernel=kernel,
                  feature=feature, share_Z=share_Z, fixed_linear_mean_function=fixed_mf,
                  q_mu=Datum.q_mu, q_sqrt=Datum.q_sqrt)
    return layer


def test_kl(reference_model):
    layer = create_layer_utility(feature=SharedIndependentMof(InducingPoints(Datum.Z)))
    layer_kl = layer.prior_kl()
    reference_kl = reference_model.prior_kl()
    np.testing.assert_allclose(layer_kl, reference_kl)


@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [True, False])
def test_predict_f(reference_model, full_cov, full_output_cov):
    layer = create_layer_utility(feature=SharedIndependentMof(InducingPoints(Datum.Z)))
    W = find_linear_mf_weights(input_dim=Datum.input_dim, output_dim=Datum.output_dim, X=Datum.X)
    layer.initialize_linear_mean_function_weights(W=W)

    X = tf.cast(tf.convert_to_tensor(Datum.X), dtype=gpflow.default_float())
    layer_mu, layer_sigma = layer.predict_f(X, full_cov, full_output_cov)
    reference_mu, reference_sigma = reference_model.predict_f(X, full_cov, full_output_cov)

    np.testing.assert_allclose(layer_mu, reference_mu)
    np.testing.assert_allclose(layer_sigma, reference_sigma)


@pytest.mark.parametrize("full_cov", [True, False])
def test_predict_f_samples(full_cov):
    num_samples = 100
    layer = create_layer_utility(feature=SharedIndependentMof(InducingPoints(Datum.Z)))
    W = find_linear_mf_weights(input_dim=Datum.input_dim, output_dim=Datum.output_dim, X=Datum.X)
    layer.initialize_linear_mean_function_weights(W=W)

    X = tf.cast(tf.convert_to_tensor(Datum.X), dtype=gpflow.default_float())
    samples, _, _ = layer.predict_f_samples(X, num_samples, full_cov)
    assert samples.shape == (num_samples, Datum.num_data, Datum.output_dim)


def test_propagate_inputs_and_features():
    layer = create_layer_utility()
    X, Z, _ = layer.propagate_inputs_and_features(Datum.X, Datum.Z)
    assert X.shape == (Datum.num_data, Datum.output_dim)
    assert Z.shape == (Datum.num_inducing, Datum.output_dim)


@pytest.mark.parametrize("feature", [None, SharedIndependentMof(InducingPoints(Datum.Z))])
@pytest.mark.parametrize("share_Z", [True, False])
def test_initialize_features(feature, share_Z):
    layer = create_layer_utility(feature=feature, share_Z=share_Z)
    if feature:
        with pytest.raises(ValueError):
            layer.initialize_features(Z=Datum.Z)

    else:
        layer.initialize_features(Z=Datum.Z)
        if share_Z:
            assert isinstance(layer.feature, gpflow.features.SharedIndependentMof)
        else:
            assert isinstance(layer.feature, gpflow.features.SeparateIndependentMof)


@pytest.mark.parametrize("fixed_mf", [True, False])
def test_initialize_mean_function_weights(fixed_mf):
    W = find_linear_mf_weights(input_dim=Datum.input_dim, output_dim=Datum.output_dim, X=Datum.X)
    layer = create_layer_utility(fixed_mf=fixed_mf)
    if fixed_mf:
        layer.initialize_linear_mean_function_weights(W=W)
        np.testing.assert_allclose(layer.mean_function.A.numpy(), W)
        assert layer.mean_function.A.trainable is False
    else:
        with pytest.raises(ValueError):
            layer.initialize_linear_mean_function_weights(W=W)


