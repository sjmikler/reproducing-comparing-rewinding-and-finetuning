# Comparing Rewinding and Fine-tuning in Neural Network Pruning -- Reproducibility Challenge 2021

### Links
* [Original paper](https://arxiv.org/abs/2003.02389)
* [Reproduction on OpenReview](https://openreview.net/forum?id=chVdm1z8sOQ)
* [Reproduction PDF](https://openreview.net/pdf?id=chVdm1z8sOQ)

### Requirements

We conducted most of our experiments using following versions of packages:

```
yaml~=0.2.5
pyyaml~=5.4.1
tensorflow~=2.4.2
tensorflow-datasets~=4.3.0
numpy~=1.19.5
tqdm~=4.61.2
```

However, different versions of TensorFlow (`2.3`, `2.5`) have been shown to work as well.
We expect readers who want to replicate our experimetns to be using a machine with at least one GPU.
Recreating them using CPU will be very time-consuming and might require some changes in code.

### Training models

Ready-to-use experiment definitions are available in `experiments` directory.
Those can be easily modified to get different experiments.
We tried to select parameter names to be self-explanatory.
You can run each experiment using `run.py` scripy with `--exp` flag.
For example:

```
python run.py --exp=experiments/resnet-20-iterative.yaml
```

If you have multiple GPU available, you can use `--gpu` flag to choose which should be used.
Otherwise, only the first GPU will be used (`/device:GPU:0`).
To use the second GPU, you can run:

```
python run.py --exp=experiments/resnet-20-one-shot.yaml --gpu=1
```

Model checkpoints will be saved under path specified in `.yaml` file, for example:

```
steps_per_epoch: 2000
save_model:
    4: data/resnet-20-iterative/baseline-ep4.h5
    36: data/resnet-20-iterative/baseline-ep36.h5
```

In this example, there will be two checkpoints: one after 8000 (2000 * 4) and second after 72000 (2000 * 36) iterations.
Those can be used as starting points for other experiments.

### Reading results

Simple logs will be available under path specified in `.yaml` file:
```
logs: data/resnet-20-iterative/info.yaml
```

More detailed logs in form of TensorBoard logs will be available under:
```
tensorboard: data/resnet-20-one-shot/tb
```
To open those, please refer to [TensorBoard](https://www.tensorflow.org/tensorboard).

