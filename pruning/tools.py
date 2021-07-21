import numpy as np
import tensorflow as tf

from pruning import sparse_layers
from training import tools


def globally_enable_pruning():
    tf.keras.layers.Dense = sparse_layers.MaskedDense
    tf.keras.layers.Conv2D = sparse_layers.MaskedConv
    print("PRUNING IS ENABLED GLOBALLY! LAYERS HAVE BEEN REPLACED...")


def structurize_saliences(saliences):
    return {k: structurize_salience(v) for k, v in saliences.items()}


def structurize_salience(saliences):
    shape = saliences.shape
    if len(shape) == 2:
        return structurize_salience_dense(saliences)
    elif len(shape) == 4:
        return structurize_salience_conv(saliences)
    else:
        raise Exception


def structurize_salience_dense(saliences):
    means = np.mean(saliences, axis=1, keepdims=True)
    return np.ones_like(saliences) * means


def structurize_salience_conv(saliences):
    shape = saliences.shape
    saliences = np.reshape(saliences, (-1, shape[-1]))
    means = np.mean(saliences, axis=0, keepdims=True)
    saliences = np.ones_like(saliences) * means
    return np.reshape(saliences, shape)


def get_pruning_mask(saliences, percentage):
    """
    :param saliences: list of saliences arrays
    :param percentage: at most this many percentage of weights will be zeroed
    :return: list of masks of 1's and 0's with corresponding sizes
    """
    sizes = [w.size for w in saliences]
    shapes = [w.shape for w in saliences]

    flatten = tools.concatenate_flattened(saliences)
    flat_mask = np.ones_like(flatten)

    threshold = np.percentile(flatten, percentage * 100)
    # print(f'pruning threshold: {threshold:8.5f}')
    flat_mask[flatten < threshold] = 0

    cumsizes = np.cumsum(sizes)[:-1]
    flat_masks = np.split(flat_mask, cumsizes)
    return [w.reshape(shape) for w, shape in zip(flat_masks, shapes)]


def saliences2masks(saliences_dict, percentage):
    """
    :param saliences_dict: keys are variable names, values are saliences
    :param percentage: float from 0 to 1
    :return: dict, keys are variable names, values are masks
    """
    saliences = list(saliences_dict.values())
    masks = get_pruning_mask(saliences, percentage)
    return {key: mask for key, mask in zip(saliences_dict, masks)}


def extract_kernels(dictionary):
    return {key: value for key, value in dictionary.items() if "kernel" in key}


def set_kernel_masks_for_model(model, masks_dict, silent=False):
    for mask in masks_dict:
        for layer in model.layers:
            for weight in layer.weights:
                if mask == weight.name:
                    layer.set_pruning_mask(masks_dict[mask])
                    if not silent:
                        print(f"{weight.name:<32} pruning to "
                              f"{layer.sparsity * 100:6.2f}%"
                              f" (left {layer.left_unpruned})")


def prune_l1(model, config, silent=False):
    """Prune smallest magnitudes."""

    sparsity = config.get('sparsity') or 0.0
    structure = config.get('structure')
    saliences = {w.name: np.abs(w.numpy()) for w in model.trainable_weights}
    saliences = extract_kernels(saliences)
    if structure:
        saliences = structurize_saliences(saliences)
    masks = saliences2masks(saliences, percentage=sparsity)
    set_kernel_masks_for_model(model, masks, silent)
    return model


def report_density(model, silent=True):
    nonzero = 0
    kernels = 0
    biases = 0
    for w in model.weights:
        if 'kernel_mask' in w.name:
            km = w.numpy()
            kernels += km.size
            nonzero_here = (km != 0).sum()
            nonzero += nonzero_here
            if not silent:
                print(f"{w.name:<32} density is {nonzero_here / km.size:6.4f}")
        if 'bias' in w.name or 'beta' in w.name:
            biases += w.shape.num_elements()
    if not silent:
        print(f"Biases make {biases / (kernels + biases) * 100:6.3f}% of weights!")
    if kernels == 0:
        return 1.0
    return nonzero / kernels


def contains_any(t, *opts):
    return any([x in t for x in opts])


def set_pruning_masks(model, pruning_method, pruning_config, dataset):
    if (pruning_method is None
            or contains_any(pruning_method.lower(), 'none', 'nothing')):
        print('NO PRUNING')
        return model
    elif contains_any(pruning_method.lower(), 'l1', 'magnitude'):
        print('WEIGHT MAGNITUDE PRUNING')
        model = prune_l1(model=model, config=pruning_config)
    else:
        raise KeyError(f"PRUNING {pruning_method} is unknown!")
    return model


def apply_pruning_for_model(model):
    """Set masked weights to 0."""

    for layer in model.layers:
        if hasattr(layer, "apply_pruning_mask"):
            layer.apply_pruning_mask()


def apply_pruning_masks(model, pruning_method):
    """Wrapper for `apply_pruning_for_model`"""
    apply_pruning_for_model(model)
    density = report_density(model, silent=True)
    print(f"REPORTING KERNELS DENSITY: {density:7.5f}")
    return model
