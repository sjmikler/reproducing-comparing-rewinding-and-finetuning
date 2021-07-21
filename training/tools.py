import os
import pickle
from collections import Counter

import numpy as np
import tensorflow as tf


def set_memory_growth():
    print("SETTING MEMORY GROWTH!")
    for gpu in tf.config.get_visible_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def set_visible_gpu(gpus=()):
    all_gpus = tf.config.get_visible_devices("GPU")
    selected_gpus = [all_gpus[idx] for idx in gpus]
    tf.config.set_visible_devices(selected_gpus, 'GPU')
    print(f"SETTING VISIBLE GPU: {selected_gpus}")


def set_precision(precision):
    import tensorflow.keras.mixed_precision.experimental as mixed_precision

    print(f"SETTING PRECISION TO {precision}")
    if precision == 16:
        policy = mixed_precision.Policy("mixed_float16")
    elif precision == 32:
        policy = mixed_precision.Policy('float32')
    elif precision == 64:
        policy = mixed_precision.Policy('float64')
    else:
        raise NameError(f"Available precision: 16, 32, 64. Not {precision}!")
    mixed_precision.set_policy(policy)


def log_from_history(history, exp):
    import datetime

    min_loss = min(history["val_loss"])
    max_acc = max(history["val_accuracy"])
    final_acc = history["val_accuracy"][-1]

    min_tr_loss = min(history["loss"])
    max_tr_acc = max(history["accuracy"])

    print(f"BEST ACCURACY: {max_acc}")

    exp["TIME"] = datetime.datetime.now().strftime("%Y.%m.%d %H:%M")
    exp["ACC"] = float(max_acc)
    exp["FINAL_ACCU"] = float(final_acc)
    exp["VALID_LOSS"] = float(min_loss)
    exp["TRAIN_ACCU"] = float(max_tr_acc)
    exp["TRAIN_LOSS"] = float(min_tr_loss)

    if exp.get('tensorboard'):
        writer = tf.summary.create_file_writer(exp['tensorboard'])
        with writer.as_default():
            for key in history:
                for idx, value in enumerate(history[key]):
                    tf.summary.scalar(key, value, idx + 1)
            tf.summary.text("experiment", data=str(exp), step=0)
    return exp


def reset_weights_to_checkpoint(model, ckp=None, skip_keyword=None):
    """Reset network in place, has an ability to skip keybword."""

    temp = tf.keras.models.clone_model(model)
    if ckp:
        temp.load_weights(ckp)
    skipped = 0
    for w1, w2 in zip(model.weights, temp.weights):
        if skip_keyword and skip_keyword in w1.name:
            skipped += 1
            continue
        w1.assign(w2)
    print(f"INFO RESET: Skipped {skipped} layers with keyword {skip_keyword}!")
    return skipped


def concatenate_flattened(arrays):
    return np.concatenate([x.flatten() if isinstance(x, np.ndarray)
                           else x.numpy().flatten() for x in arrays], axis=0)


def print_model_info(model):
    print(f"MODEL INFO")
    layer_counts = Counter()
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer_counts['Dense'] += 1
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer_counts['Conv2D'] += 1
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer_counts['BatchNorm'] += 1
        if isinstance(layer, tf.keras.layers.Dropout):
            layer_counts['Dropout'] += 1
    print(f"LAYER COUNTS: {dict(layer_counts)}")

    bn = 0
    biases = 0
    kernels = 0
    trainable_w = 0
    for w in model.trainable_weights:
        n = w.shape.num_elements()
        trainable_w += n

    for layer in model.layers:
        if hasattr(layer, 'beta') and layer.beta is not None:
            bn += layer.beta.shape.num_elements()

        if hasattr(layer, 'gamma') and layer.gamma is not None:
            bn += layer.gamma.shape.num_elements()

        if hasattr(layer, 'bias') and layer.bias is not None:
            biases += layer.bias.shape.num_elements()

        if hasattr(layer, 'kernel'):
            kernels += layer.kernel.shape.num_elements()

    print(f"TRAINABLE WEIGHTS: {trainable_w}")
    print(f"KERNELS: {kernels} ({kernels / trainable_w * 100:^6.2f}%), "
          f"BIASES: {biases} ({biases / trainable_w * 100:^6.2f}%), "
          f"BN: {bn} ({bn / trainable_w * 100:^6.2f}%)")


def save_optimizer(optimizer, path):
    if dirpath := os.path.dirname(path):
        os.makedirs(dirpath, exist_ok=True)
    weights = optimizer.get_weights()
    with open(path, 'wb') as f:
        pickle.dump(weights, f)


def save_model(model, path):
    if dirpath := os.path.dirname(path):
        os.makedirs(dirpath, exist_ok=True)
    model.save_weights(path, save_format="h5")


class CheckpointAfterEpoch(tf.keras.callbacks.Callback):
    def __init__(self, epoch2path=None, epoch2path_optim=None):
        super().__init__()
        if epoch2path is None:
            epoch2path = {}
        if epoch2path_optim is None:
            epoch2path_optim = {}

        self.epoch2path = epoch2path
        self.epoch2path_optim = epoch2path_optim
        self.created_model_ckp = []
        self.created_optim_ckp = []

    def on_epoch_end(self, epoch, logs=None):
        next_epoch = epoch + 1

        if next_epoch in self.epoch2path:
            path = self.epoch2path[next_epoch]
            save_model(self.model, path)
            self.created_model_ckp.append(path)

        if next_epoch in self.epoch2path_optim:
            path = self.epoch2path_optim[next_epoch]
            save_optimizer(self.model.optimizer, path)
            self.created_optim_ckp.append(path)

    def list_created_checkpoints(self):
        print(f"CREATED MODEL CHECKPOINTS:")
        for ckp in self.created_model_ckp:
            print(ckp)
        if not self.created_model_ckp:
            print("NONE")
        print(f"CREATED OPTIM CHECKPOINTS:")
        for ckp in self.created_optim_ckp:
            print(ckp)
        if not self.created_optim_ckp:
            print("NONE")


def get_optimizer_lr_metric(opt):
    if hasattr(opt, '_decayed_lr'):
        def lr(*args):
            return opt._decayed_lr(tf.float32)

        return lr
    else:
        return None
