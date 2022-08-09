# Adapted from https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import RandomTranslation,RandomCrop,RandomFlip,RandomZoom
from aconnect.layers import Conv_AConnect, FC_AConnect, DepthWiseConv_AConnect

kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

def conv3x3(x, out_planes, stride=1,**AConnect_args):
    x = layers.ZeroPadding2D(padding=1)(x)
    x = Conv_AConnect(filters=out_planes, kernel_size=(3,3), strides=stride, 
                    **AConnect_args)(x)
    return x 

def basic_block(x, planes, stride=1, downsample=None,**AConnect_args):
    identity = x

    out = conv3x3(x, planes, stride=stride,**AConnect_args)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)

    out = conv3x3(out, planes,**AConnect_args)
    #out = layers.BatchNormalization()(out) # Removed by Luis E. Rueda G.

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add()([identity, out])
    out = layers.BatchNormalization()(out)  # Added by Luis E. Rueda G.
    out = layers.ReLU()(out)

    return out

def make_layer(x, planes, blocks, stride=1,**AConnect_args):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            Conv_AConnect(filters=planes, kernel_size=(1,1), strides=stride,
                    **AConnect_args),
            #layers.BatchNormalization(),   # Removed by Luis E. Rueda G.
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
    if Wstd!=0:
        x = RandomZoom(0.0)(inputs)
        x = RandomTranslation(0.0,0.0)(x)
        x = RandomZoom(0.0)(x)
    else:
        Flip = RandomFlip("horizontal")
        x = Flip(inputs)
        x = RandomTranslation(0.1,0.1)(x)
        x = RandomZoom(0.2)(x)
    
    #x = layers.ZeroPadding2D(padding=3)(x)
    #x = Conv_AConnect(filters=64, kernel_size=(7,7), strides=2,
    x = Conv_AConnect(filters=64, kernel_size=(3,3), strides=1,
                    **AConnect_args)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    #x = layers.ZeroPadding2D(padding=1)(x)
    #x = layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = make_layer(x, 64, blocks_per_layer[0],**AConnect_args)
    x = make_layer(x, 128, blocks_per_layer[1], stride=2,**AConnect_args)
    x = make_layer(x, 256, blocks_per_layer[2], stride=2,**AConnect_args)
    x = make_layer(x, 512, blocks_per_layer[3], stride=2,**AConnect_args)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = FC_AConnect(units=num_classes,
                    **AConnect_args)(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    model.summary
    return model

def resnet18(input_shape, **kwargs):
    return resnet(input_shape, [2, 2, 2, 2], **kwargs)

def resnet34(input_shape, **kwargs):
    return resnet(input_shape, [3, 4, 6, 3], **kwargs)
