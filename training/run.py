import tensorflow as tf

import pruning.tools
import training.tools
from training import datasets, models


def run_experiment(exp):
    """Train as specified in experiment dict.

    PROCEDURES IN ORDER:
    1. Creating dataset, model, optimizer
    2. Loading checkpoint Before Pruning
    3. Applying pruning
    4. Loading checkpoint After Pruning
    5. Pruning related procedures After Pruning
    6. Training
    """

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=exp["lr_boundaries"],
            values=exp["lr_values"],
        ),
        momentum=0.9,
        nesterov=True,
    )

    pruning.tools.globally_enable_pruning()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    dataset = datasets.cifar(version=exp["cifar_version"])
    model = models.ResNetStiff(**exp["model_args"])

    lr_metric = training.tools.get_optimizer_lr_metric(optimizer)
    metrics = ["accuracy", lr_metric]
    model.compile(optimizer, loss_fn, metrics=metrics)
    training.tools.print_model_info(model)

    # load checkpointed all weights before the pruning
    if hasattr(exp, 'load_model_before_pruning') and exp.load_model_before_pruning:
        model.load_weights(exp.load_model_before_pruning)
        print(f"LOADED BEFORE PRUNING {exp.load_model_before_pruning}")

    model = pruning.tools.set_pruning_masks(model=model,
                                            pruning_method=exp["pruning"],
                                            pruning_config=exp.get("pruning_config"),
                                            dataset=dataset)
    assert isinstance(model, tf.keras.Model)

    # load or reset weights after the pruning, do not change masks
    if 'load_model_after_pruning' in exp and exp['load_model_after_pruning']:
        path = exp['load_model_after_pruning']
        if path == 'random':
            ckp = None
        else:
            ckp = path
        num_masks = training.tools.reset_weights_to_checkpoint(
            model,
            ckp=ckp,
            skip_keyword='kernel_mask')
        print(f"LOADED AFTER PRUNING {path}, but keeping {num_masks} masks")

    checkpoint_callback = training.tools.CheckpointAfterEpoch(
        epoch2path=exp['save_model'])

    # just apply pruning by zeroing weights with previously calculated masks
    pruning.tools.apply_pruning_masks(model, pruning_method=exp['pruning'])

    checkpoint_callback.set_model(model)
    checkpoint_callback.on_epoch_end(epoch=-1)  # for checkpointing before training
    callbacks = [checkpoint_callback]

    if exp['epochs'] > exp['initial_epoch']:
        history = model.fit(x=dataset['train'],
                            validation_data=dataset['test'],
                            steps_per_epoch=exp['steps_per_epoch'],
                            epochs=exp['epochs'],
                            initial_epoch=exp['initial_epoch'],
                            callbacks=callbacks).history

        exp["FINAL_DENSITY"] = pruning.tools.report_density(model)
        print("FINAL DENSITY:", exp["FINAL_DENSITY"])
        training.tools.log_from_history(history, exp=exp)
    checkpoint_callback.list_created_checkpoints()
