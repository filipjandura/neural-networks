import tensorflow as tf
from tensorflow import keras

kl = keras.layers
kb = keras.backend
ki = keras.initializers


class Network:
    def __init__(self, classes, batch_size=36, img_size=32, channels=3):
        self.classes = classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.channels = channels

    def loss(self):
        return keras.losses.categorical_crossentropy

    def build_network(self, params):
        pass

    def conv_bn_sc_relu(self, x, filters, kernel, stride, padding='same', drop_rate=0.3,):
        conv = kl.Conv2D(filters, kernel, stride, padding,
                         kernel_initializer=ki.truncated_normal,
                         bias_initializer=ki.random_normal,
                         )(x)
        bnorm = kl.BatchNormalization()(conv)
        dropout = kl.Dropout(rate=drop_rate)(bnorm)
        relu = kl.ReLU()(dropout)

        return relu

    def saf_pool(self, x, kernel, stride, padding='same'):
        pool = kl.MaxPool2D(kernel, stride, padding)(x)
        dropout = kl.Dropout(rate=0.1)(pool)

        return dropout

    def residual(self, x, filters, kernel, stride=(1, 1), padding='same', _shortcut=False):
        shortcut = x

        conv1 = kl.Conv2D(filters, kernel_size=kernel, strides=stride, padding=padding)(x)
        bnorm = kl.BatchNormalization()(conv1)
        lrelu = kl.LeakyReLU()(bnorm)

        conv2 = kl.Conv2D(filters, kernel_size=kernel, strides=(1, 1), padding=padding)(lrelu)
        out_ = kl.BatchNormalization()(conv2)

        if _shortcut or stride != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = kl.Conv2D(filters, (1, 1), strides=stride, padding=padding)(shortcut)
            shortcut = kl.BatchNormalization()(shortcut)

            out_ = kl.add([shortcut, out_])
            out_ = kl.LeakyReLU()(out_)

        return out_