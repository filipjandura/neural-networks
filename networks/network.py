from tensorflow import keras
import tensorflow as tf

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