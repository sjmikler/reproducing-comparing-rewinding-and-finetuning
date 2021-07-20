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


def mnist(train_batch_size=100,
          valid_batch_size=400,
          dtype=tf.float32,
          shuffle_train=10000,
          data_dir=None):
    def preprocess(x, y):
        x = tf.cast(x, dtype)
        x /= 255
        return x, y

    ds = tfds.load(name='mnist', as_supervised=True, data_dir=data_dir)
    ds['train'] = ds['train'].repeat()
    ds['train'] = ds['train'].shuffle(shuffle_train)
    ds['train'] = ds['train'].map(preprocess)
    ds['train'] = ds['train'].batch(train_batch_size)

    ds['test'] = ds['test'].map(preprocess)
    ds['test'] = ds['test'].batch(valid_batch_size)

    ds['input_shape'] = (28, 28, 1)
    ds['n_classes'] = 10
    return ds


def test(train_batch_size=100,
         image_shape=(32, 32, 3),
         dtype=tf.float32):
    images = tf.ones([2, *image_shape])
    target = tf.constant([0, 1])

    def preprocess(x, y):
        x = tf.cast(x, dtype)
        return x, y

    ds = dict()
    ds['train'] = tf.data.Dataset.from_tensor_slices((images, target))
    ds['train'] = ds['train'].map(preprocess).repeat().batch(train_batch_size)
    ds['test'] = tf.data.Dataset.from_tensor_slices((images, target))
    ds['test'] = ds['test'].map(preprocess).batch(2)
    return ds


def get_dataset_from_alias(alias, precision=32):
    assert isinstance(alias, str)

    if precision == 16:
        dtype = tf.float16
    elif precision == 32:
        dtype = tf.float32
    elif precision == 64:
        dtype = tf.float64
    else:
        raise NotImplementedError(f"Unknown precision {precision}!")

    if alias == 'cifar10':
        return cifar(dtype=dtype, version=10)
    elif alias == 'cifar100':
        return cifar(dtype=dtype, version=100)
    elif alias == 'mnist':
        return mnist(dtype=dtype)
    else:
        raise NotImplementedError(f"Unknown alias {alias}")


def figure_out_input_shape(ds):
    for x, y in ds['test']:
        break
    else:
        raise RuntimeError("Dataset is empty!")
    return x.shape[1:]


def figure_out_n_classes(ds):
    classes = set()
    for x, y in ds['test']:
        classes.update(y.numpy())
    return len(classes)
