import numpy as np


def find_linear_mf_weights(input_dim, output_dim, X):
    """
    Find the initial weights of the Linear mean function based on
    input and output dimensions of the layer

    :param input_dim: input dimension of layer
    :param output_dim: output dimension of layer
    :param X: numpy array of data with which mean function is initialized
    """
    assert X.shape[1] - input_dim in [0, 1]

    if input_dim <= output_dim:
        W = np.eye(input_dim, output_dim)

    elif input_dim > output_dim:
        # Slicing to guard against index column at the end
        _, _, V = np.linalg.svd(X[:, :input_dim], full_matrices=False)
        W = V[:output_dim, :].T

    if X.shape[1] == input_dim + 1:  # Accounting for a potential index column
        W = np.concatenate([W, np.zeros((1, W.shape[1]))], axis=0)

    return W
