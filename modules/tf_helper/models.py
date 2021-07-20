from collections.abc import Iterable
import re

import tensorflow as tf

try:
    from ._initialize import *
except ImportError:
    pass

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class GemPool(tf.keras.layers.Layer):
    def __init__(self, pool_size=None, initial_value=3.0):
        super().__init__()
        self.initial_value = initial_value
        self.pool = pool_size

    def call(self, flow, **kwds):
        input_dtype = flow.dtype
        flow = tf.cast(flow, tf.float32)
        flow = tf.clip_by_value(flow, 1e-6, 1e3)
        self.p.assign(tf.clip_by_value(self.p, 1, 6))

        flow = tf.pow(flow, self.p)

        if self.pool:
            flow = tf.keras.layers.AvgPool2D(
                self.pool, padding="same", dtype="float32"
            )(flow)
        else:
            flow = tf.keras.layers.GlobalAvgPool2D(dtype="float32")(flow)

        flow = tf.pow(flow, tf.divide(1.0, self.p))
        return tf.cast(flow, input_dtype)

    def build(self, *args):
        self.p = tf.Variable(
            self.initial_value, trainable=True, name="gempool_p", dtype="float32"
        )
        super().build(*args)


def classifier(
    flow,
    n_classes,
    regularizer=None,
    bias_regularizer=None,
    initializer="glorot_uniform",
    pooling="avgpool",
):
    if pooling == "catpool":
        maxp = tf.keras.layers.GlobalMaxPool2D()(flow)
        avgp = tf.keras.layers.GlobalAvgPool2D()(flow)
        flow = tf.keras.layers.Concatenate()([maxp, avgp])
    if pooling == "avgpool":
        flow = tf.keras.layers.GlobalAvgPool2D()(flow)
    if pooling == "maxpool":
        flow = tf.keras.layers.GlobalMaxPool2D()(flow)
    if pooling == "gempool":
        flow = GemPool(initial_value=3.0)(flow)

    # multiple-head version
    if isinstance(n_classes, Iterable):
        outs = [
            tf.keras.layers.Dense(
                n_class,
                bias_regularizer=bias_regularizer,
                kernel_regularizer=regularizer,
                kernel_initializer=initializer,
            )(flow)
            for n_class in n_classes
        ]
    else:
        outs = tf.keras.layers.Dense(
            n_classes,
            bias_regularizer=bias_regularizer,
            kernel_regularizer=regularizer,
            kernel_initializer=initializer,
        )(flow)
    return outs


def VGG(
    input_shape,
    n_classes,
    version=None,
    l1_reg=0,
    l2_reg=1e-4,
    group_sizes=(1, 1, 2, 2, 2),
    features=(64, 128, 256, 512, 512),
    pools=(2, 2, 2, 2, 2),
    regularize_bias=True,
    **kwds,
):
    print(f"VGG: unknown parameters: {list(kwds)}")
    if version:
        if version == 11:
            group_sizes = (1, 1, 2, 2, 2)
        elif version == 13:
            group_sizes = (2, 2, 2, 2, 2)
        elif version == 16:
            group_sizes = (2, 2, 3, 3, 3)
        elif version == 19:
            group_sizes = (2, 2, 4, 4, 4)
        else:
            raise KeyError(f"Unkown version={version}!")

    regularizer = (
        tf.keras.regularizers.l1_l2(l1_reg, l2_reg) if l2_reg or l1_reg else None
    )
    bias_regularizer = regularizer if regularize_bias else None

    def conv3x3(*args, **kwds):
        # bias is not needed, since batch norm does it
        return tf.keras.layers.Conv2D(
            *args,
            **kwds,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizer,
        )

    def bn_relu(x):
        x = tf.keras.layers.BatchNormalization(
            beta_regularizer=bias_regularizer, gamma_regularizer=bias_regularizer,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        )(x)
        return tf.keras.layers.ReLU()(x)

    inputs = tf.keras.layers.Input(shape=input_shape)
    flow = inputs

    skip_first_maxpool = True
    for group_size, width, pool in zip(group_sizes, features, pools):

        if not skip_first_maxpool:
            flow = tf.keras.layers.MaxPool2D(pool)(flow)
        else:
            skip_first_maxpool = False

        for _ in range(group_size):
            flow = conv3x3(filters=width)(flow)
            flow = bn_relu(flow)

    outs = classifier(
        flow, n_classes, regularizer=regularizer, bias_regularizer=bias_regularizer
    )
    model = tf.keras.Model(inputs=inputs, outputs=outs)
    return model


def ResNet(
    dataset=None,
    input_shape=None,
    n_classes=None,
    version=None,
    l1_reg=0,
    l2_reg=2e-4,
    bootleneck=False,
    strides=(1, 2, 2),
    group_sizes=(2, 2, 2),
    features=(16, 32, 64),
    initializer="he_uniform",
    activation="tf.nn.relu",
    final_pooling="avgpool",
    dropout=0,
    regularize_bias=True,
    # preactivate_blocks=True,
    resnet_version=2,
    shortcut_projection=True,
    head=(("conv", 16, 3, 1),),
    **kwds,
):
    if dataset == 'cifar' or dataset == 'cifar10':
        input_shape = (32, 32, 3)
        n_classes = 10
    elif dataset == 'cifar100':
        input_shape = (32, 32, 3)
        n_classes = 100
    elif dataset == 'mnist':
        input_shape = (28, 28, 1)
        n_classes = 10
    else:
        assert input_shape is not None
        assert n_classes is not None

    if version == 20:
        group_sizes = (3, 3, 3)
    elif version == 56:
        group_sizes = (9, 9, 9)
    elif version == 110:
        group_sizes = (18, 18, 18)
    elif isinstance(version, str) and 'WRN' in version:
        if '-' in version:
            version = (int(x) for x in version.strip('WRN').strip('-').split('-'))
        elif '_' in version:
            version = (int(x) for x in version.strip('WRN').strip('_').split('_'))
        else:
            raise KeyError("WRN{N}-{K} is the proper format!")
        N, K = version
        assert (N - 4) % 6 == 0
        size = int((N - 4) / 6)
        group_sizes = (size, size, size),
        features = (16 * K, 32 * K, 64 * K),
    elif version is not None:
        raise KeyError(f"Version {version} unknown!")
    print(f"ResNet: unknown parameters: {list(kwds)}")

    activation_func = eval(activation)
    if l2_reg or l1_reg:
        regularizer = tf.keras.regularizers.l1_l2(l1_reg, l2_reg)
    else:
        regularizer = None
    bias_regularizer = regularizer if regularize_bias else None

    def conv(filters, kernel_size, use_bias=False, **kwds):
        return tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            bias_regularizer=bias_regularizer,
            **kwds,
        )

    def shortcut(x, filters, strides):
        if not shortcut_projection:
            m_filters = filters - x.shape[-1]
            m_width = x.shape[1] // strides
            m_height = x.shape[2] // strides
            return tf.pad(
                x[:, :m_width, :m_height], [[0, 0], [0, 0], [0, 0], [m_filters, 0]]
            )
        else:
            return tf.keras.layers.Conv2D(
                filters,
                kernel_size=1,
                use_bias=False,
                strides=strides,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
            )(x)

    def bn_activate(x, remove_relu=False):
        x = tf.keras.layers.BatchNormalization(
            beta_regularizer=bias_regularizer, gamma_regularizer=bias_regularizer,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        )(x)
        return x if remove_relu else activation_func(x)

    def simple_block2(flow, filters, strides):
        projection_shortcut = flow.shape[-1] != filters or strides != 1
        flow_shortcut = flow
        flow = bn_activate(flow)
        if projection_shortcut:
            flow_shortcut = shortcut(flow, filters, strides)

        flow = conv(filters, 3, strides=strides)(flow)
        flow = bn_activate(flow)

        if dropout:
            flow = tf.keras.layers.Dropout(dropout)(flow)
        flow = conv(filters, 3, strides=1)(flow)
        return flow + flow_shortcut

    def simple_block1(flow, filters, strides):
        projection_shortcut = flow.shape[-1] != filters or strides != 1
        flow_shortcut = flow

        if projection_shortcut:
            flow_shortcut = shortcut(flow, filters, strides)
            flow_shortcut = bn_activate(flow_shortcut, remove_relu=True)

        flow = conv(filters, 3, strides=strides)(flow)
        flow = bn_activate(flow)

        if dropout:
            flow = tf.keras.layers.Dropout(dropout)(flow)
        flow = conv(filters, 3, strides=1)(flow)
        flow = bn_activate(flow, remove_relu=True)
        return activation_func(flow + flow_shortcut)

    if bootleneck:
        raise NotImplementedError
    elif resnet_version == 2:
        block = simple_block2
    elif resnet_version == 1:
        block = simple_block1
    else:
        raise NotImplementedError

    inputs = tf.keras.Input(input_shape)
    flow = inputs

    # BUILDING HEAD OF THE NETWORK
    for name, *args in head:
        if name == "conv":
            bias = True if "bias" in args else False
            flow = conv(args[0], args[1], strides=args[2], use_bias=bias)(flow)
        if name == "maxpool":
            flow = tf.keras.layers.MaxPool2D(args[0])(flow)
        if name == "avgpool":
            flow = tf.keras.layers.AvgPool2D(args[0])(flow)
        if name == "relu":
            flow = tf.nn.relu(flow)

    if resnet_version == 2:
        flow = bn_activate(flow)

    # BUILD THE RESIDUAL BLOCKS
    for group_size, width, stride in zip(group_sizes, features, strides):
        for _ in range(group_size):
            flow = block(flow, width, stride)
            stride = 1  # reset stride for the rest of the group

    # BUILDING THE CLASSIFIER
    if resnet_version == 2:
        flow = bn_activate(flow, remove_relu=True)
        flow = tf.nn.relu(flow)
        # use relu here even if different activation is choosen
        # this is for consistency in classifier inputs being non-negative

    outputs = classifier(
        flow,
        n_classes,
        regularizer=regularizer,
        bias_regularizer=bias_regularizer,
        initializer=initializer,
        pooling=final_pooling,
    )
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def ResNetStiff(
    dataset=None,
    alias=None,
    input_shape=None,
    n_classes=None,
    resnet_version=2,
    features=(16, 32, 64),
    l1_reg=0,
    l2_reg=2e-4,
    initializer="he_uniform",
    activation="tf.nn.relu",
    BLOCKS_IN_GROUP=3,
    BATCH_NORM_DECAY=0.997,  # 0.9
    BATCH_NORM_EPSILON=1e-5,  # 1e-3
    final_pooling="avgpool",
    dropout=0,
    regularize_bias=True,
    shortcut_conv_projection=True,
):
    if dataset == 'cifar' or dataset == 'cifar10':
        input_shape = (32, 32, 3)
        n_classes = 10
    elif dataset == 'cifar100':
        input_shape = (32, 32, 3)
        n_classes = 100
    elif dataset == 'mnist':
        input_shape = (28, 28, 1)
        n_classes = 10
    else:
        assert input_shape is not None
        assert n_classes is not None

    if alias == 'WRN-16-8':
        N = 16
        K = 8
        assert (N - 4) % 6 == 0
        size = int((N - 4) / 6)
        BLOCKS_IN_GROUP = size
        features = (16 * K, 32 * K, 64 * K),

    activation_func = eval(activation)
    if l2_reg or l1_reg:
        regularizer = tf.keras.regularizers.l1_l2(l1_reg, l2_reg)
    else:
        regularizer = None
    bias_regularizer = regularizer if regularize_bias else None

    def conv(filters, kernel_size, use_bias=False, **kwds):
        return tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            bias_regularizer=bias_regularizer,
            **kwds,
        )

    def shortcut(x, filters, strides):
        if not shortcut_conv_projection:
            m_filters = filters - x.shape[-1]
            m_width = x.shape[1] // strides
            m_height = x.shape[2] // strides
            return tf.pad(
                x[:, :m_width, :m_height], [[0, 0], [0, 0], [0, 0], [m_filters, 0]]
            )
        else:
            return tf.keras.layers.Conv2D(
                filters,
                kernel_size=1,
                use_bias=False,
                strides=strides,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
            )(x)

    def bn_activate(x, remove_relu=False):
        x = tf.keras.layers.BatchNormalization(
            beta_regularizer=bias_regularizer, gamma_regularizer=bias_regularizer,
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
        )(x)
        return x if remove_relu else activation_func(x)

    def simple_block2(flow,
                      filters,
                      strides,
                      activate_shortcut=False):
        flow_shortcut = flow
        flow = bn_activate(flow)
        if activate_shortcut:
            flow_shortcut = flow
        if flow.shape[-1] != filters or strides != 1:
            flow_shortcut = shortcut(flow_shortcut, filters, strides)

        flow = conv(filters, 3, strides=strides)(flow)
        flow = bn_activate(flow)

        if dropout:
            flow = tf.keras.layers.Dropout(dropout)(flow)
        flow = conv(filters, 3, strides=1)(flow)
        return flow + flow_shortcut

    def simple_block1(flow, filters, strides):
        flow_shortcut = flow
        if flow.shape[-1] != filters or strides != 1:
            flow_shortcut = shortcut(flow, filters, strides)
            flow_shortcut = bn_activate(flow_shortcut, remove_relu=True)

        flow = conv(filters, 3, strides=strides)(flow)
        flow = bn_activate(flow)

        if dropout:
            flow = tf.keras.layers.Dropout(dropout)(flow)
        flow = conv(filters, 3, strides=1)(flow)
        flow = bn_activate(flow, remove_relu=True)
        return activation_func(flow + flow_shortcut)

    inputs = tf.keras.Input(input_shape)
    flow = inputs

    flow = conv(16, 3, strides=1, use_bias=False)(flow)

    if resnet_version == 2:
        flow = simple_block2(flow,
                             filters=features[0],
                             strides=1,
                             activate_shortcut=True)
        for _ in range(BLOCKS_IN_GROUP - 1):
            flow = simple_block2(flow, filters=features[0], strides=1)

        flow = simple_block2(flow,
                             filters=features[1],
                             strides=2,
                             activate_shortcut=True)
        for _ in range(BLOCKS_IN_GROUP - 1):
            flow = simple_block2(flow, filters=features[1], strides=1)

        flow = simple_block2(flow,
                             filters=features[2],
                             strides=2,
                             activate_shortcut=True)
        for _ in range(BLOCKS_IN_GROUP - 1):
            flow = simple_block2(flow, filters=features[2], strides=1)

        flow = bn_activate(flow, remove_relu=True)
        flow = tf.nn.relu(flow)

    elif resnet_version == 1:
        flow = bn_activate(flow)

        flow = simple_block1(flow, filters=features[0], strides=1)
        for _ in range(BLOCKS_IN_GROUP - 1):
            flow = simple_block1(flow, filters=features[0], strides=1)

        flow = simple_block1(flow, filters=features[1], strides=2)
        for _ in range(BLOCKS_IN_GROUP - 1):
            flow = simple_block1(flow, filters=features[1], strides=1)

        flow = simple_block1(flow, filters=features[2], strides=2)
        for _ in range(BLOCKS_IN_GROUP - 1):
            flow = simple_block1(flow, filters=features[2], strides=1)

    outputs = classifier(
        flow,
        n_classes,
        regularizer=regularizer,
        bias_regularizer=bias_regularizer,
        initializer=initializer,
        pooling=final_pooling,
    )
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def LeNet(
    input_shape,
    n_classes,
    l1_reg=0,
    l2_reg=0,
    layer_sizes=(300, 100),
    initializer="glorot_uniform",
    **kwds,
):
    print(f"LeNet: unknown parameters: {list(kwds)}")
    regularizer = (
        tf.keras.regularizers.l1_l2(l1_reg, l2_reg) if l2_reg or l1_reg else None
    )
    initializer = initializer

    def dense(*args, **kwds):
        return tf.keras.layers.Dense(
            *args,
            **kwds,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
        )

    inputs = tf.keras.layers.Input(shape=input_shape)

    flow = tf.keras.layers.Flatten()(inputs)
    for layer_size in layer_sizes:
        flow = dense(layer_size, activation="relu")(flow)

    outs = dense(n_classes, activation=None)(flow)
    model = tf.keras.Model(inputs=inputs, outputs=outs)
    return model


def LeNetConv(
    input_shape, n_classes, l1_reg=0, l2_reg=0, initializer="glorot_uniform", **kwds
):
    print(f"Unknown parameters: {list(kwds)}")
    regularizer = (
        tf.keras.regularizers.l1_l2(l1_reg, l2_reg) if l2_reg or l1_reg else None
    )
    initializer = initializer

    def dense(*args, **kwds):
        return tf.keras.layers.Dense(
            *args,
            **kwds,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
        )

    def conv(*args, **kwds):
        return tf.keras.layers.Conv2D(
            *args,
            **kwds,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
        )

    inputs = tf.keras.layers.Input(shape=input_shape)

    flow = conv(20, 5, activation="relu")(inputs)
    flow = tf.keras.layers.MaxPool2D(2)(flow)
    flow = conv(50, 5, activation="relu")(flow)
    flow = tf.keras.layers.MaxPool2D(2)(flow)

    flow = tf.keras.layers.Flatten()(flow)
    flow = dense(500, activation="relu")(flow)

    outs = dense(n_classes, activation=None)(flow)
    model = tf.keras.Model(inputs=inputs, outputs=outs)
    return model


def get_model_from_alias(alias, input_shape, n_classes):
    assert isinstance(alias, str)

    if m := re.match(r"WRN(\d+)-(\d+)", alias):
        N, K = m.groups()
        return ResNet(input_shape=input_shape,
                      n_classes=n_classes,
                      version=f"WRN{N}-{K}")
    elif m := re.match(r"VGG(\d+)", alias):
        (N,) = m.groups()
        return VGG(input_shape=input_shape, n_classes=n_classes, version=int(N))

    else:
        raise NotImplementedError(f"Unknown alias {alias}")
