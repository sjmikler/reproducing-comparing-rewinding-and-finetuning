"""Source created in notebook: notebooks\training_functools.ipynb"""

from collections import defaultdict
from itertools import islice

import tensorflow as tf
from tqdm import tqdm


@tf.function
def train_step(x, y, model):
    assert isinstance(model, tf.keras.Model)
    assert model.optimizer is not None, "Model not compiled!"
    assert model.loss is not None, "Model not compiled!"
    mixed_precision = isinstance(
        model.optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer
    )
    with tf.GradientTape() as tape:
        outs = model(x, training=True)
        outs = tf.cast(outs, tf.float32)
        loss = model.compiled_loss(y, outs, regularization_losses=model.losses)
        if mixed_precision:
            loss = model.optimizer.get_scaled_loss(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    if mixed_precision:
        gradients = model.optimizer.get_unscaled_gradients(gradients)

    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    model.compiled_metrics.update_state(y, outs)
    return outs


# %%


def train_epoch(iterator, model, epoch_idx=0, steps=None, callbacks=(), use_pbar=None):
    for callback in callbacks:
        assert isinstance(callback, tf.keras.callbacks.Callback)
        callback.on_epoch_begin(epoch_idx)

    if use_pbar:
        pbar = use_pbar
    else:
        pbar = tqdm(total=steps, leave=True, ascii=True)

    for bidx, (x, y) in enumerate(islice(iterator, steps)):
        for callback in callbacks:
            assert isinstance(callback, tf.keras.callbacks.Callback)
            callback.on_train_batch_begin(bidx)

        outs = train_step(x, y, model)
        pbar.set_postfix(
            {m.name: m.result().numpy() for m in model.metrics}, refresh=False
        )

        for callback in callbacks:
            assert isinstance(callback, tf.keras.callbacks.Callback)
            callback.on_train_batch_end(bidx)
        pbar.update()

    if not use_pbar:
        pbar.close()

    for callback in callbacks:
        assert isinstance(callback, tf.keras.callbacks.Callback)
        callback.on_epoch_end(epoch_idx)


# %%


@tf.function
def valid_step(x, y, model):
    assert isinstance(model, tf.keras.Model)
    assert model.loss is not None, "Model not compiled!"

    outs = model(x, training=False)
    outs = tf.cast(outs, tf.float32)
    model.compiled_loss(y, outs)
    model.compiled_metrics.update_state(y, outs)
    return outs


def valid_epoch(iterator, model, epoch_idx=0, steps=None, callbacks=()):
    for bidx, (x, y) in enumerate(islice(iterator, steps)):
        for callback in callbacks:
            assert isinstance(callback, tf.keras.callbacks.Callback)
            callback.on_test_batch_begin(bidx)

        outs = valid_step(x, y, model)

        for callback in callbacks:
            assert isinstance(callback, tf.keras.callbacks.Callback)
            callback.on_test_batch_end(bidx)


# %%


def reset_metrics(model):
    results = {m.name: m.result().numpy() for m in model.metrics}
    for metric in model.metrics:
        metric.reset_states()
    return results


def fit(
    model,
    training_data,
    validation_data,
    steps_per_epoch=None,
    epochs=1,
    initial_epoch=0,
    callbacks=(),
):
    history = defaultdict(list)

    bpbar = tqdm(total=epochs, leave=True, ascii=True)
    for epoch_idx in range(initial_epoch, epochs):
        pbar = tqdm(total=steps_per_epoch, leave=True, ascii=True)
        train_epoch(
            training_data,
            model,
            epoch_idx=epoch_idx,
            steps=steps_per_epoch,
            callbacks=callbacks,
            use_pbar=pbar,
        )
        metrics = reset_metrics(model)
        for key, value in metrics.items():
            history[key].append(value)

        valid_epoch(
            validation_data,
            model,
            epoch_idx=epoch_idx,
            steps=steps_per_epoch,
            callbacks=callbacks,
        )
        metrics = reset_metrics(model)
        for key, value in metrics.items():
            history["val_" + key].append(value)

        metrics = reset_metrics(model)

        pbar.close()
        bpbar.set_postfix({key: value[-1] for key, value in history.items()})
        bpbar.update()
    return dict(history)


# %%
