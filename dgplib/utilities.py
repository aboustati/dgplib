from typing import Optional

from tabulate import tabulate
import numpy as np
import tensorflow as tf

from gpflow.base import Parameter
from gpflow.config import summary_fmt
from gpflow.utilities.utilities import _str_tensor_value, _merge_leaf_components


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


def print_summary(module: tf.Module, fmt: str = None):
    """
    Prints a summary of the parameters and variables contained in a tf.Module.
    """

    fmt = fmt if fmt is not None else summary_fmt()
    column_names = ['name', 'class', 'transform', 'trainable', 'shape', 'dtype', 'value']

    def get_name(v):
        return v.__class__.__name__

    def get_transform(v):
        if hasattr(v, "transform") and v.transform is not None:
            return v.transform.__class__.__name__
        return None

    merged_leaf_components = _merge_leaf_components(leaf_components(module))

    column_values = [[
        path,
        get_name(variable),
        get_transform(variable),
        variable.trainable,
        variable.shape,
        variable.dtype.name,
        _str_tensor_value(variable.numpy())
    ] for path, variable in merged_leaf_components.items()]

    if fmt == "notebook":
        from IPython.core.display import display, HTML
        tab = tabulate(column_values, headers=column_names, tablefmt="html")
        display(HTML(tab))
    else:
        print(tabulate(column_values, headers=column_names, tablefmt=fmt))


def leaf_components(input: tf.Module):
    return _get_leaf_components(input)


def _get_leaf_components(input: tf.Module, prefix: Optional[str] = None):
    """
    Returns a list of tuples each corresponding to a gpflow.Parameter or tf.Variable in the each
    submodules of a given tf.Module. Each tuple consists of an specific Parameter (or Variable) and
    its relative path inside the module, which is constructed recursively by adding a prefix with
    the path to the current module. Designed to be used as a helper for the method 'print_summary'.
    :param module: tf.Module including keras.Model, keras.layers.Layer and gpflow.Module.
    :param prefix: string containing the relative path to module, by default set to None.
    :return:
    """
    if not isinstance(input, tf.Module):
        raise TypeError("Input object expected to have `tf.Module` type")

    prefix = input.__class__.__name__ if prefix is None else prefix
    var_dict = dict()

    for key, submodule in vars(input).items():
        if key in tf.Module._TF_MODULE_IGNORED_PROPERTIES:
            continue
        elif isinstance(submodule, Parameter) or isinstance(submodule, tf.Variable):
            var_dict[f"{prefix}.{key}"] = submodule
        elif isinstance(submodule, tf.Module):
            submodule_var = _get_leaf_components(submodule, prefix=f"{prefix}.{key}")
            var_dict.update(submodule_var)
        elif isinstance(submodule, list) and isinstance(submodule[0], tf.Module):
            for i, component in enumerate(submodule):
                submodule_var = _get_leaf_components(component, prefix=f"{prefix}.{key}[{i}]")
                var_dict.update(submodule_var)

    return var_dict

