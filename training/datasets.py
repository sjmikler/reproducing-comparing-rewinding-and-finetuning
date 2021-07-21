import tensorflow as tf
import tensorflow_datasets as tfds


def cifar(train_batch_size=128,
          valid_batch_size=512,
          padding='reflect',
          dtype=tf.float32,
          shuffle_train=20000,
          repeat_train=True,
          version=10,
          data_dir=None):
    subtract = tf.constant([0.49139968, 0.48215841, 0.44653091], dtype=dtype)
    divide = tf.constant([0.24703223, 0.24348513, 0.26158784], dtype=dtype)

    def train_prep(x, y):
        x = tf.cast(x, dtype) / 255.0
        x = tf.image.random_flip_left_right(x)
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]], mode=padding)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = (x - subtract) / divide
        return x, y

    def valid_prep(x, y):
        x = tf.cast(x, dtype) / 255.0
        x = (x - subtract) / divide
        return x, y

    if version == 10 or version == 100:
        ds = tfds.load(name=f'cifar{version}', as_supervised=True, data_dir=data_dir)
    else:
        raise Exception(f"version = {version}, but should be either 10 or 100!")

    if repeat_train:
        ds['train'] = ds['train'].repeat()
    if shuffle_train:
        ds['train'] = ds['train'].shuffle(shuffle_train)
    ds['train'] = ds['train'].map(train_prep)
    ds['train'] = ds['train'].batch(train_batch_size)
    # ds['train'] = ds['train'].prefetch(tf.data.experimental.AUTOTUNE)

    ds['test'] = ds['test'].map(valid_prep)
    ds['test'] = ds['test'].batch(valid_batch_size)
    return ds
