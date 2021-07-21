# Comparing Rewinding and Fine-tuning in Neural Network Pruning -- Reproducibility Challenge 2021

### Requirements

Most of our experiments were conducted using following versions of packages:

```
yaml~=0.2.5
pyyaml~=5.4.1
tensorflow~=2.4.2
tensorflow-datasets~=4.3.0
numpy~=1.19.5
tqdm~=4.61.2
```

However, different versions of TensorFlow have been shown to work as well.

### Training models

Ready-to-use experiment definitions are available in `experiments` directory. You can run each of them using `run.py` scripy with `--exp` flag. For example:

```
python run.py --exp=experiments/resnet-20-iterative.yaml
```

If you have multiple GPU available, you can use `--gpu` flag to choose which should be used:
For example:

```
python run.py --exp=experiments/resnet-20-one-shot.yaml --gpu=1
```

Weights will be saved in a place specified in `.yaml` file, for example:

```
save_model:
    4: data/resnet-20-iterative/baseline-ep4.h5
    36: data/resnet-20-iterative/baseline-ep36.h5
```

In this example, there will be two checkpoints, one after 4th and second after 36th epoch. Those can be used as starting points for other experiments.

### Reading results

Simple logs will be available under path specified in `.yaml` file:

```
logs: data/resnet-20-iterative/info.yaml
```

More detailed logs in form of TensorBoard logs will be available in:
```
...
```

