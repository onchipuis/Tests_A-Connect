import tensorflow as tf
import math
from aconnect.layers import Conv_AConnect, FC_AConnect, DepthWiseConv_AConnect

NUM_CLASSES = 10


def swish(x):
    return x * tf.nn.sigmoid(x)


def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25,**AConnect_args):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = Conv_AConnect(filters=self.num_reduced_filters,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same",
                                                  **AConnect_args)
        self.expand_conv = Conv_AConnect(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same",
                                                  **AConnect_args)

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        output = inputs * branch
        return output


class MBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k,
            drop_connect_rate,**AConnect_args):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = Conv_AConnect(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            **AConnect_args)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dwconv = DepthWiseConv_AConnect(kernel_size=(k, k),
                                            strides=stride,
                                            padding="same",
                                            **AConnect_args)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(input_channels=in_channels * expansion_factor,
                                            **AConnect_args)
        self.conv2 = Conv_AConnect(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            **AConnect_args)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=drop_connect_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        x = swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = self.dropout(x, training=training)
            x = tf.keras.layers.add([x, inputs])
        return x


def build_mbconv_block(in_channels, out_channels, layers, stride,
        expansion_factor, k, drop_connect_rate,**AConnect_args):
    block = tf.keras.Sequential()
    for i in range(layers):
        if i == 0:
            block.add(MBConv(in_channels=in_channels,
                            out_channels=out_channels,
                            expansion_factor=expansion_factor,
                            stride=stride,
                            k=k,
                            drop_connect_rate=drop_connect_rate,
                            **AConnect_args))
        else:
            block.add(MBConv(in_channels=out_channels,
                            out_channels=out_channels,
                            expansion_factor=expansion_factor,
                            stride=1,
                            k=k,
                            drop_connect_rate=drop_connect_rate,
                            **AConnect_args))
    return block


class EfficientNet(tf.keras.Model):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate,
            drop_connect_rate=0.2,
            Wstd=0,Bstd=0,
            isQuant=["no","no"],bw=[8,8],
            Conv_pool=2,FC_pool=2,errDistr="normal",
            bwErrProp=True,**kwargs):
        super(EfficientNet, self).__init__()

        AConnect_args = {"Wstd": Wstd,
                        "Bstd": Bstd,
                        "isQuant": isQuant,
                        "bw": bw,
                        "errDistr": errDistr,
                        "bwErrProp": bwErrProp,
                        "d_type": tf.dtypes.float16}
        
        self.conv1 = Conv_AConnect(filters=round_filters(32, width_coefficient),
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same",
                                            pool=Conv_pool,
                                            **AConnect_args)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.block1 = build_mbconv_block(in_channels=round_filters(32, width_coefficient),
                                        out_channels=round_filters(16, width_coefficient),
                                        layers=round_repeats(1, depth_coefficient),
                                        stride=1,
                                        expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate,
                                        pool=Conv_pool,
                                        **AConnect_args)
        self.block2 = build_mbconv_block(in_channels=round_filters(16, width_coefficient),
                                        out_channels=round_filters(24, width_coefficient),
                                        layers=round_repeats(2, depth_coefficient),
                                        stride=2,
                                        expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate,
                                        pool=Conv_pool,
                                        **AConnect_args)
        self.block3 = build_mbconv_block(in_channels=round_filters(24, width_coefficient),
                                        out_channels=round_filters(40, width_coefficient),
                                        layers=round_repeats(2, depth_coefficient),
                                        stride=2,
                                        expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate,
                                        pool=Conv_pool,
                                        **AConnect_args)
        self.block4 = build_mbconv_block(in_channels=round_filters(40, width_coefficient),
                                        out_channels=round_filters(80, width_coefficient),
                                        layers=round_repeats(3, depth_coefficient),
                                        stride=2,
                                        expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate,
                                        pool=Conv_pool,
                                        **AConnect_args)
        self.block5 = build_mbconv_block(in_channels=round_filters(80, width_coefficient),
                                        out_channels=round_filters(112, width_coefficient),
                                        layers=round_repeats(3, depth_coefficient),
                                        stride=1,
                                        expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate,
                                        pool=Conv_pool,
                                        **AConnect_args)
        self.block6 = build_mbconv_block(in_channels=round_filters(112, width_coefficient),
                                        out_channels=round_filters(192, width_coefficient),
                                        layers=round_repeats(4, depth_coefficient),
                                        stride=2,
                                        expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate,
                                        pool=Conv_pool,
                                        **AConnect_args)
        self.block7 = build_mbconv_block(in_channels=round_filters(192, width_coefficient),
                                        out_channels=round_filters(320, width_coefficient),
                                        layers=round_repeats(1, depth_coefficient),
                                        stride=1,
                                        expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate,
                                        pool=Conv_pool,
                                        **AConnect_args)

        self.conv2 = Conv_AConnect(filters=round_filters(1280, width_coefficient),
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            pool=Conv_pool,
                                            **AConnect_args)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.fc = FC_AConnect(units=NUM_CLASSES,pool=FC_pool,**AConnect_args)
        self.act = tf.keras.layers.Activation('softmax') 

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = swish(x)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)
        x = self.act(x)

        return x


def model_creation(width_coefficient, depth_coefficient, resolution, dropout_rate):
    model = EfficientNet(width_coefficient=width_coefficient,
                       depth_coefficient=depth_coefficient,
                       dropout_rate=dropout_rate,**AConnect_args)
    model.build(input_shape=(None, resolution, resolution, 3))
    #model.summary()

    return model


def efficient_net_b0(**AConnect_args):
    return model_creation(1.0, 1.0, 224, 0.2,**AConnect_args)


def efficient_net_b1(**AConnect_args):
    return model_creation(1.0, 1.1, 240, 0.2,**AConnect_args)


def efficient_net_b2(**AConnect_args):
    return model_creation(1.1, 1.2, 260, 0.3,**AConnect_args)


def efficient_net_b3(**AConnect_args):
    return model_creation(1.2, 1.4, 300, 0.3,**AConnect_args)


def efficient_net_b4(**AConnect_args):
    return model_creation(1.4, 1.8, 380, 0.4,**AConnect_args)


def efficient_net_b5(**AConnect_args):
    return model_creation(1.6, 2.2, 456, 0.4,**AConnect_args)


def efficient_net_b6(**AConnect_args):
    return model_creation(1.8, 2.6, 528, 0.5,**AConnect_args)


def efficient_net_b7(**AConnect_args):
    return model_creation(2.0, 3.1, 600, 0.5,**AConnect_args)

