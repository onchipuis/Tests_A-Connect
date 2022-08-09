# Adapted from https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from aconnect.layers import Conv_AConnect, FC_AConnect, DepthWiseConv_AConnect

kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

def conv3x3(x, out_planes, stride=1,**AConnect_args):
    x = layers.ZeroPadding2D(padding=1)(x)
    x = Conv_AConnect(filters=out_planes, kernel_size=(3,3), strides=stride, 
                    **AConnect_args)(x)
                    #kernel_initializer=kaiming_normal,
    return x 

def basic_block(x, planes, stride=1, downsample=None,**AConnect_args):
    identity = x

    out = conv3x3(x, planes, stride=stride,**AConnect_args)
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(out)
    out = layers.ReLU()(out)

    out = conv3x3(out, planes,**AConnect_args)
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add()([identity, out])
    out = layers.ReLU()(out)

    return out

def make_layer(x, planes, blocks, stride=1,**AConnect_args):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            Conv_AConnect(filters=planes, kernel_size=(1,1), strides=stride,
                    **AConnect_args),
                    #kernel_initializer=kaiming_normal,
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
        ]

    x = basic_block(x, planes, stride, downsample)
    for i in range(1, blocks):
        x = basic_block(x, planes)

    return x

def resnet(input_shape, blocks_per_layer, num_classes=100,
            Wstd=0,Bstd=0,
            isQuant=["no","no"],bw=[8,8],
            Conv_pool=8,FC_pool=8,errDistr="normal",
            bwErrProp=True,**kwargs):
    
    AConnect_args = {"Wstd": Wstd,
                    "Bstd": Bstd,
                    "isQuant": isQuant,
                    "bw": bw,
                    "errDistr": errDistr,
                    "bwErrProp": bwErrProp,
                    "d_type": tf.dtypes.float16}
    
    inputs = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=3)(inputs)
    x = Conv_AConnect(filters=64, kernel_size=(7,7), strides=2,
                    **AConnect_args)(x)
                    #kernel_initializer=kaiming_normal,
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = make_layer(x, 64, blocks_per_layer[0],**AConnect_args)
    x = make_layer(x, 128, blocks_per_layer[1], stride=2,**AConnect_args)
    x = make_layer(x, 256, blocks_per_layer[2], stride=2,**AConnect_args)
    x = make_layer(x, 512, blocks_per_layer[3], stride=2,**AConnect_args)

    x = layers.GlobalAveragePooling2D()(x)
    #initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    x = FC_AConnect(units=num_classes,
                    **AConnect_args)(x)
                    #kernel_initializer=initializer, 
                    #bias_initializer=initializer,**AConnect_args)(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=x)
    return model

def resnet18(input_shape, **kwargs):
    return resnet(input_shape, [2, 2, 2, 2], **kwargs)

def resnet34(input_shape, **kwargs):
    return resnet(input_shape, [3, 4, 6, 3], **kwargs)
