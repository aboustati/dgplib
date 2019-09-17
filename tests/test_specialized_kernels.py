import pytest

import numpy as np
import tensorflow as tf

from dgplib.specialized_kernels import SwitchedKernel

import gpflow
from gpflow.kernels import Kernel
from gpflow.base import Parameter


rng = np.random.RandomState(42)


class DummyKernel(Kernel):
    def __init__(self, val, active_dims=None, name=None):
        super().__init__(active_dims, name)
        self.val = Parameter(val, dtype=np.float32)

    def K(self, X, Y=None, presliced=False):
        if Y is None:
            Y = X
        return self.val * tf.ones((tf.shape(X)[0], tf.shape(Y)[0]))

    def K_diag(self, X, presliced=False):
        return self.val * tf.ones((tf.shape(X)[0], 1))


@pytest.fixture
def X1():
    X1_ind = np.array([0, 1, 2, 2, 1, 0, 1, 0, 2, 1])[:, None]
    X1 = np.hstack([rng.randn(10, 3), X1_ind]).astype(gpflow.default_float())
    return X1


@pytest.fixture
def X2():
    X2_ind = np.array([0, 1, 2, 2, 1])[:, None]
    X2 = np.hstack([rng.randn(5, 3), X2_ind]).astype(gpflow.default_float())
    return X2


@pytest.fixture
def kernel():
    K1 = DummyKernel(1.0)
    K2 = DummyKernel(2.0)
    K3 = DummyKernel(3.0)
    kern_list = [K1, K2, K3]
    kernel = SwitchedKernel(kern_list, 3)
    return kernel


def test_K(kernel, X1, X2):
    reference = np.array([
        [1, 0, 0, 0, 0],
        [0, 2, 0, 0, 2],
        [0, 0, 3, 3, 0],
        [0, 0, 3, 3, 0],
        [0, 2, 0, 0, 2],
        [1, 0, 0, 0, 0],
        [0, 2, 0, 0, 2],
        [1, 0, 0, 0, 0],
        [0, 0, 3, 3, 0],
        [0, 2, 0, 0, 2]
    ])

    gram_matrix = kernel.K(X1, X2)
    np.testing.assert_allclose(gram_matrix, reference)


def test_Kdiag(kernel, X1):
    gram_matrix = kernel.K_diag(X1)
    np.testing.assert_allclose(gram_matrix, X1[:, -1:] + 1)
