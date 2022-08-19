# Copyright 2019 The TensorFlow Authors, Tung Shu-Cheng. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Taken from https://github.com/GdoongMathew/EfficientNetV2
import string
import math
import collections

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils
from aconnect.layers import Conv_AConnect, FC_AConnect, DepthWiseConv_AConnect

#from .config import *
#from .utils import CONV_KERNEL_INITIALIZER
#from .utils import DENSE_KERNEL_INITIALIZER
#from .utils import round_filters
#from .utils import round_repeats

#from .weights import IMAGENET_WEIGHTS_URL, WEIGHTS_MAP

####################### UTILS #################################
"""
This is from https://github.com/qubvel/efficientnet/blob/master/efficientnet/model.py
"""

def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

####################### CONFIG #################################
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

V2_BASE_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=32,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=32, output_filters=48,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=48, output_filters=96,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=5, input_filters=96, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=8, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
]

V2_S_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=24, output_filters=24,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=4, input_filters=24, output_filters=48,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=4, input_filters=48, output_filters=64,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=6, input_filters=64, output_filters=128,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=9, input_filters=128, output_filters=160,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=15, input_filters=160, output_filters=256,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
]

V2_M_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=24, output_filters=24,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=5, input_filters=24, output_filters=48,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=5, input_filters=48, output_filters=80,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=7, input_filters=80, output_filters=160,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=14, input_filters=160, output_filters=176,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=18, input_filters=176, output_filters=304,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=5, input_filters=304, output_filters=512,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
]

V2_L_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=4, input_filters=32, output_filters=32,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=7, input_filters=32, output_filters=64,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=7, input_filters=64, output_filters=96,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=10, input_filters=96, output_filters=192,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=19, input_filters=192, output_filters=224,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=25, input_filters=224, output_filters=384,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=7, input_filters=384, output_filters=640,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
]

V2_XL_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=4, input_filters=32, output_filters=32,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=8, input_filters=32, output_filters=64,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=8, input_filters=64, output_filters=96,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=16, input_filters=96, output_filters=192,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=24, input_filters=192, output_filters=256,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=32, input_filters=256, output_filters=512,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=8, input_filters=512, output_filters=640,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
]

####################### CONFIG #################################
IMAGENET_WEIGHTS_URL = 'https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/'

WEIGHTS_MAP = {
    'efficientnetv2_s': {
        'imagenet':         {
            True:   'efficientnetv2-s_imagenet_top.h5',
            False:  'efficientnetv2-s_imagenet_notop.h5',
        },
        'imagenet21k':      {
            False:  'efficientnetv2-s_imagenet21k_notop.h5',
        },
        'imagenet21k-ft1k': {
            True:   'efficientnetv2-s_imagenet21k-ft1k_top.h5',
            False:  'efficientnetv2-s_imagenet21k-ft1k_notop.h5',
        },
    },
    'efficientnetv2_m': {
        'imagenet':         {
            True:   'efficientnetv2-m_imagenet_top.h5',
            False:  'efficientnetv2-m_imagenet_notop.h5',
        },
        'imagenet21k':      {
            False:  'efficientnetv2-m_imagenet21k_notop.h5',
        },
        'imagenet21k-ft1k': {
            True:   'efficientnetv2-m_imagenet21k-ft1k_top.h5',
            False:  'efficientnetv2-m_imagenet21k-ft1k_notop.h5',
        },
    },
    'efficientnetv2_l': {
        'imagenet':         {
            True:   'efficientnetv2-l_imagenet_top.h5',
            False:  'efficientnetv2-l_imagenet_notop.h5',
        },
        'imagenet21k':      {
            False:  'efficientnetv2-l_imagenet21k_notop.h5',
        },
        'imagenet21k-ft1k': {
            True:   'efficientnetv2-l_imagenet21k-ft1k_top.h5',
            False:  'efficientnetv2-l_imagenet21k-ft1k_notop.h5',
        },
    },
    'efficientnetv2_xl': {
        'imagenet21k':      {
            False:  'efficientnetv2-xl_imagenet21k_notop.h5',
        },
        'imagenet21k-ft1k': {
            True:   'efficientnetv2-xl_imagenet21k-ft1k_top.h5',
            False:  'efficientnetv2-xl_imagenet21k-ft1k_notop.h5',
        },
    }
}


####################### MODELS #################################
def get_dropout():
    """Wrapper over custom dropout. Fix problem of ``None`` shape for tf.keras.
    It is not possible to define FixedDropout class as global object,
    because we do not have modules for inheritance at first time.

    Issue:
        https://github.com/tensorflow/tensorflow/issues/30946
    """

    class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    return FixedDropout


def mb_conv_block(inputs,
                  block_args: BlockArgs,
                  activation='swish',
                  drop_rate=None,
                  prefix='',
                  conv_dropout=None,
                  mb_type='normal',
                  **AConnect_args):
    """Fused Mobile Inverted Residual Bottleneck"""
    assert mb_type in ['normal', 'fused']
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    Dropout = get_dropout()

    x = inputs

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = Conv_AConnect(filters,
                          1 if mb_type == 'normal' else block_args.kernel_size,
                          strides=1 if mb_type == 'normal' else block_args.strides,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          padding="SAME",
                          use_bias=False,
                          name=f'{prefix}expand_conv',
                          **AConnect_args)(x)
        x = layers.BatchNormalization(axis=bn_axis, name=f'{prefix}expand_bn')(x)
        x = layers.Activation(activation=activation, name=f'{prefix}expand_activation')(x)

    if mb_type is 'normal':
        x = DepthWiseConv_AConnect(block_args.kernel_size,
                                   block_args.strides,
                                   depthwise_initializer=CONV_KERNEL_INITIALIZER,
                                   padding="SAME",
                                   use_bias=False,
                                   name=f'{prefix}dwconv',
                                   **AConnect_args)(x)
        x = layers.BatchNormalization(axis=bn_axis, name=f'{prefix}bn')(x)
        x = layers.Activation(activation=activation, name=f'{prefix}activation')(x)

    if conv_dropout and block_args.expand_ratio > 1:
        x = layers.Dropout(conv_dropout)(x)

    if has_se:
        num_reduced_filters = max(1, int(
            block_args.input_filters * block_args.se_ratio
        ))
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        target_shape = (1, 1, filters) if backend.image_data_format() == 'channels_last' else (filters, 1, 1)
        se_tensor = layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = Conv_AConnect(num_reduced_filters, 1,
                                  activation=activation,
                                  padding="SAME",
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_reduce',
                                  **AConnect_args)(se_tensor)
        se_tensor = Conv_AConnect(filters, 1,
                                  activation='sigmoid',
                                  padding="SAME",
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_expand',
                                  **AConnect_args)(se_tensor)
        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = Conv_AConnect(block_args.output_filters,
                      kernel_size=1 if block_args.expand_ratio != 1 else block_args.kernel_size,
                      strides=1 if block_args.expand_ratio != 1 else block_args.strides,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      padding="SAME",
                      use_bias=False,
                      name=f'{prefix}project_conv',
                      **AConnect_args)(x)

    x = layers.BatchNormalization(axis=bn_axis, name=f'{prefix}project_bn')(x)
    if block_args.expand_ratio == 1:
        x = layers.Activation(activation=activation, name=f'{prefix}activation')(x)

    if all(s == 1 for s in block_args.strides) \
            and block_args.input_filters == block_args.output_filters:
        if drop_rate and drop_rate > 0:
            x = Dropout(drop_rate,
                        noise_shape=(None, 1, 1, 1),
                        name=f'{prefix}dropout')(x)
        x = layers.Add(name=f'{prefix}add')([x, inputs])
    return x


def EfficientNetV2(blocks_args,
                   width_coefficient,
                   depth_coefficient,
                   default_resolution,
                   arch,
                   dropout_rate=0.2,
                   depth_divisor=8,
                   model_name='efficientnetv2',
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   activation='swish',
                   pooling='avg',
                   final_drop_rate=None,
                   num_classes=1000,
                   Wstd=0,Bstd=0,
                   isQuant=["no","no"],bw=[8,8],
                   Conv_pool=8,FC_pool=8,errDistr="normal",
                   bwErrProp=True,
                   **kwargs):
    """
    Create an EfficientNetV2 model using given inputs.
    Will reload pretrained weights if provided.
    :param blocks_args: a list of BlockArgs objects, specifying model's configuration.
    :param width_coefficient: float, width scaling coefficient.
    :param depth_coefficient: float, depth scaling coefficient.
    :param default_resolution: integer, default input size.
    :param arch: string, model architect
    :param dropout_rate: float, dropout rate
    :param depth_divisor: int.
    :param model_name: string, model name.
    :param include_top: bool, whether to add the final classification layers.
    :param weights: string or None.
    :param input_tensor: (optional) tensorflow keras tensor, used as the inputs if given.
    :param input_shape: (optional) input shape.
    :param activation: string, activation type, default to `swish`
    :param pooling: (optional) pooling mode in feature extraction.
        - `avg`: global average pooling.
        - `max`: global maximum pooling.
    :param final_drop_rate: dropout rate before the final output layer if include_top is set to True
    :param num_classes: (optional) number of num_classes to in the final output.
    :param kwargs:
    :return: tf.keras Model instance.
    """
    AConnect_args = {"Wstd": Wstd,
                    "Bstd": Bstd,
                    "isQuant": isQuant,
                    "bw": bw,
                    "errDistr": errDistr,
                    "bwErrProp": bwErrProp,
                    "d_type": tf.dtypes.float16}

    assert isinstance(blocks_args, list) and False not in [isinstance(block_args, BlockArgs) for block_args in
                                                           blocks_args]
    assert pooling in ['avg', 'max', None]

    input_shape = (default_resolution, default_resolution, 3) if input_shape is None else input_shape

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if backend.backend() == 'tensorflow':
            from tensorflow.keras.backend import is_keras_tensor
        else:
            is_keras_tensor = backend.is_keras_tensor
        if not is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    inputs = img_input if input_tensor is None else keras_utils.get_source_inputs(input_tensor)

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    # build stem layer
    x = img_input

    x = Conv_AConnect(round_filters(blocks_args[0].input_filters, width_coefficient, depth_divisor), 3,
                      strides=2,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      use_bias=False,
                      padding="SAME",
                      name='stem_conv',
                      pool=Conv_pool,
                      **AConnect_args)(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation=activation, name='stem_activation')(x)

    mb_type = {
        0: 'normal',
        1: 'fused'
    }

    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0

    # build blocks
    for idx, block_args in enumerate(blocks_args):
        assert isinstance(block_args, BlockArgs)
        assert block_args.num_repeat > 0
        input_filters = round_filters(block_args.input_filters,
                                      width_coefficient,
                                      depth_divisor)
        output_filters = round_filters(block_args.output_filters,
                                       width_coefficient,
                                       depth_divisor)
        repeats = round_repeats(block_args.num_repeat, depth_coefficient)

        block_args = block_args._replace(
            input_filters=input_filters,
            output_filters=output_filters,
            num_repeat=repeats
        )
        drop_rate = dropout_rate * float(block_num) / num_blocks_total

        conv_type = mb_type[block_args.conv_type]
        x = mb_conv_block(x, block_args,
                          activation=activation,
                          drop_rate=drop_rate,
                          mb_type=conv_type,
                          prefix=f'{conv_type}_block{idx + 1}a_',
                          pool=Conv_pool,
                          **AConnect_args)
        block_num += 1
        if block_args.num_repeat > 1:
            block_args = block_args._replace(
                input_filters=block_args.output_filters,
                strides=[1, 1]
            )
            for _idx in range(block_args.num_repeat - 1):
                drop_rate = dropout_rate * float(block_num) / num_blocks_total
                block_prefix = f'{conv_type}_block{idx + 1}{string.ascii_lowercase[_idx + 1]}_' if \
                    _idx + 1 < len(string.ascii_lowercase) else \
                    f'{conv_type}_block{idx + 1}{string.ascii_uppercase[_idx + 1 - len(string.ascii_lowercase)]}_'
                x = mb_conv_block(x, block_args,
                                  activation=activation,
                                  drop_rate=drop_rate,
                                  mb_type=conv_type,
                                  prefix=block_prefix,
                                  pool=Conv_pool,
                                  **AConnect_args)
                block_num += 1

    # build head
    x = Conv_AConnect(
        filters=round_filters(1280, width_coefficient, depth_divisor),
        kernel_size=1,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="SAME",
        use_bias=False,
        name='head_conv',
        pool=Conv_pool,
        **AConnect_args)(x)
    x = layers.BatchNormalization(axis=bn_axis, name='head_bn')(x)
    x = layers.Activation(activation=activation, name='head_activation')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='head_avg_pool')(x)
        if final_drop_rate and final_drop_rate > 0:
            x = layers.Dropout(final_drop_rate, name='head_dropout')(x)
        x = FC_AConnect(num_classes,
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name='probs',
                         pool=FC_pool,
                         **AConnect_args)(x)
        outputs = layers.Softmax()(x)
    else:
        if pooling == 'avg':
            outputs = layers.GlobalAveragePooling2D(name='head_avg_pool')(x)
        elif pooling == 'max':
            outputs = layers.GlobalMaxPooling2D(name='head_max_pool')(x)

    model = models.Model(inputs=inputs,outputs=outputs,name=model_name)


    if weights in ['imagenet', 'imagenet21k', 'imagenet21k-ft1k']:
        file_name = WEIGHTS_MAP[arch][weights][include_top]
        weight_path = keras_utils.get_file(
            file_name,
            IMAGENET_WEIGHTS_URL + file_name,
            cache_subdir='models'
        )
        model.load_weights(weight_path)
    elif weights:
        model.load_weights(weights)

    # Included by Luis E. Rueda G.
    if not(include_top):
        if input_shape[1]==32:
            x = tf.keras.layers.experimental.preprocessing.Resizing(128,128)(img_input)
            x = model.layers[1](x)
        x = FC_AConnect(num_classes,
                     kernel_initializer=DENSE_KERNEL_INITIALIZER,
                     name='probs',
                     pool=FC_pool,
                     **AConnect_args)(model.layers[-1].output)
        outputs = layers.Softmax()(x)
        model = models.Model(inputs=inputs,outputs=outputs,name=model_name)

    model.summary()
    return model


def EfficientNetV2_Base(include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        num_classes=1000,
                        **kwargs
                        ):
    return EfficientNetV2(
        V2_BASE_BLOCKS_ARGS,
        1., 1., 300, 'efficientnetv2_base',
        dropout_rate=0.2,
        model_name='efficientnetv2_base',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        num_classes=num_classes,
        **kwargs
    )


def EfficientNetV2_S(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     num_classes=1000,
                     **kwargs
                     ):
    return EfficientNetV2(
        V2_S_BLOCKS_ARGS,
        1., 1., 300, 'efficientnetv2_s',
        dropout_rate=0.2,
        model_name='efficientnetv2_s',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        num_classes=num_classes,
        **kwargs
    )


def EfficientNetV2_M(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     num_classes=1000,
                     **kwargs
                     ):
    return EfficientNetV2(
        V2_M_BLOCKS_ARGS,
        1., 1., 384, 'efficientnetv2_m',
        dropout_rate=0.2,
        model_name='efficientnetv2_m',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        num_classes=num_classes,
        **kwargs
    )


def EfficientNetV2_L(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     num_classes=1000,
                     **kwargs
                     ):
    return EfficientNetV2(
        V2_L_BLOCKS_ARGS,
        1., 1., 384, 'efficientnetv2_l',
        dropout_rate=0.4,
        model_name='efficientnetv2_l',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        num_classes=num_classes,
        **kwargs
    )


def EfficientNetV2_XL(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      num_classes=1000,
                      **kwargs
                      ):
    return EfficientNetV2(
        V2_XL_BLOCKS_ARGS,
        1., 1., 384, 'efficientnetv2_xl',
        dropout_rate=0.4,
        model_name='efficientnetv2_xl',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        num_classes=num_classes,
        **kwargs
    )


setattr(EfficientNetV2_Base, '__doc__', EfficientNetV2.__doc__)
setattr(EfficientNetV2_S, '__doc__', EfficientNetV2.__doc__)
setattr(EfficientNetV2_M, '__doc__', EfficientNetV2.__doc__)
setattr(EfficientNetV2_L, '__doc__', EfficientNetV2.__doc__)
setattr(EfficientNetV2_XL, '__doc__', EfficientNetV2.__doc__)
