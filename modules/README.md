# tf-helper

**Available command line arguments** 

```
optional arguments:
  -h, --help          show this help message and exit
  --gpu GPU           Which GPUs to use during training, e.g. 0,1,3 or 1
  --no-memory-growth  Disables memory growth
```

**Available experiment parameters** (required*)

* `precision`* is either 16, 32 or 64

# pruning

Inherits from: **tf-helper**. You can use arguments and parameters from there.

**Available experiment parameters** (required*)

* `steps`*
* `steps_per_epoch`*
* `initial_epoch` might be useful for resuming training
* `load_model_before_pruning` loads full model from given path, including kernel masks
* `load_model_after_pruning` loads model, but skips kernel masks. This is used for some pruning methods.
* `load_optimizer` load optimizer states from a checkpoint
* `pruning`* and `pruning_config`*
* `dataset`* and `dataset_config`*
* `model`* and `model_config`*
* `tensorboard_log`
* `save_model`* is a dictionary with keys being epoch numbers, e.g. `{16: path_to_model_after_16.h5}`
* `save_optim`* e.g. `{16: path_to_optimizer_after_16.h5, 32: path_to_optimizer_after_32.pkl}`

**Fun Facts**

* Tensorboard logs with training and validation history are saved all at once, after the training in `experiment.yaml/tensorboard_log`. Nothing will be saved if training is interrupted.
