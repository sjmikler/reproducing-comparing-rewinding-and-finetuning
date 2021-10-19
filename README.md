# Comparing Rewinding and Fine-tuning in Neural Network Pruning: Reproducibility Challenge 2021

### Links
* [Original paper](https://arxiv.org/abs/2003.02389)
* [Reproduction PDF](ANON_REPORT.pdf)

### Requirements

We conducted most of our experiments using the following versions of packages:

```
yaml~=0.2.5
pyyaml~=5.4.1
tensorflow~=2.4.2
tensorflow-datasets~=4.3.0
numpy~=1.19.5
tqdm~=4.61.2
```

However, different versions of TensorFlow (`2.3`, `2.5`) have been shown to work as well.
We expect readers who want to replicate our experiments to be using a machine with at least one GPU.
Recreating them using CPU will be very time-consuming and might require some changes in code.

### Training models

Ready-to-use experiment definitions are available in the `experiments` directory.
That can be easily modified to get different experiments.
We tried to select parameter names to be self-explanatory.
You can run each experiment using the `run.py` script with `--exp` flag.
For example:

```
python run.py --exp=experiments/resnet-20-iterative.yaml
```

If you have multiple GPUs available, you can use `--gpu` flag to choose which one should be used.
Otherwise, the default is the first listed GPU (usually `/device:GPU:0`).
GPU indexing starts at 0.
To use the second GPU, you can run:

```
python run.py --exp=experiments/resnet-20-one-shot.yaml --gpu=1
```

Model checkpoints will be saved under the path specified in `.yaml` file, for example:

```
steps_per_epoch: 2000
save_model:
    4: data/resnet-20-iterative/baseline-ep4.h5
    36: data/resnet-20-iterative/baseline-ep36.h5
```

In this example, there will be two checkpoints: one after 8000 (2000 * 4) and the second after 72000 (2000 * 36) iterations.
Those can be used as starting points for other experiments.

### Reading results

Simple logs will be available under the path specified in `.yaml` file:
```
logs: data/resnet-20-iterative/info.yaml
```

More detailed logs in the form of TensorBoard logs will be available under:
```
tensorboard: data/resnet-20-one-shot/tb
```

To open those, please refer to [TensorBoard](https://www.tensorflow.org/tensorboard).


### Checkpoints of already trained models

We can upload some checkpoints via Zenodo platform.
Please contact us directly.
