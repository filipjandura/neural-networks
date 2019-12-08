from networks.network import Network
from tensorflow import keras
import tensorflow as tf


kl = keras.layers
kb = keras.backend


class SimpleNet_v2(Network):
    def __init__(self, classes, batch_size=36, img_size=32, channels=3):
        Network.__init__(self, classes=classes, batch_size=batch_size, img_size=img_size, channels=channels)

    def build_network(self, params):
        input = kl.Input(shape=(self.img_size, self.img_size, self.channels))
        conv1 = self.conv_bn_sc_relu(input, 66, kernel=(3,3), stride=2, padding='same', drop_rate=params.drop_rate)
        conv2 = self.conv_bn_sc_relu(conv1, 128, kernel=(3,3), stride=1, padding='same', drop_rate=params.drop_rate)
        conv3 = self.conv_bn_sc_relu(conv2, 128, kernel=(3,3), stride=1, padding='same', drop_rate=params.drop_rate)
        conv4 = self.conv_bn_sc_relu(conv3, 128, kernel=(3,3), stride=1, padding='same', drop_rate=params.drop_rate)
        conv5 = self.conv_bn_sc_relu(conv4, 192, kernel=(3,3), stride=1, padding='same', drop_rate=params.drop_rate)
        pool1 = self.saf_pool(conv5, kernel=2, stride=2, padding='valid')
        conv6 = self.conv_bn_sc_relu(pool1, 192, kernel=(3,3), stride=1, padding='same', drop_rate=params.drop_rate)
        conv7 = self.conv_bn_sc_relu(conv6, 192, kernel=(3,3), stride=1, padding='same', drop_rate=params.drop_rate)
        conv8 = self.conv_bn_sc_relu(conv7, 192, kernel=(3,3), stride=1, padding='same', drop_rate=params.drop_rate)
        conv9 = self.conv_bn_sc_relu(conv8, 128, kernel=(3,3), stride=1, padding='same', drop_rate=params.drop_rate)
        conv10 = self.conv_bn_sc_relu(conv9, 288, kernel=(3,3), stride=1, padding='same', drop_rate=params.drop_rate)
        pool2 = self.saf_pool(conv10, kernel=2, stride=2, padding='valid')
        conv11 = self.conv_bn_sc_relu(pool2, 288, kernel=(3,3), stride=1, padding='same', drop_rate=params.drop_rate)
        conv12 = self.conv_bn_sc_relu(conv11, 355, kernel=(3,3), stride=1, padding='same', drop_rate=params.drop_rate)
        conv13 = self.conv_bn_sc_relu(conv12, 432, kernel=(3, 3), stride=1, padding='same', drop_rate=params.drop_rate)
        gmpool = self.saf_pool(conv13, 2, stride=2, padding='valid')
        flatten = kl.Flatten()(gmpool)
        logits = kl.Dense(self.classes)(flatten)

        self.ios = [input, logits]
        self.model = keras.models.Model(input, logits, name='SimpleNet_v2')
        self.model.summary()

    def loss(self):
        def loss_(y_true, y_pred):
            return keras.losses.categorical_crossentropy(y_true, y_pred, True)
        return loss_


class SimpleNet_residual(Network):
    def __init__(self, classes, batch_size=36, img_size=32, channels=3):
        Network.__init__(self, classes=classes, batch_size=batch_size, img_size=img_size, channels=channels)

    def build_network(self, params):
        input = kl.Input(shape=(self.img_size, self.img_size, self.channels))
        conv1 = self.conv_bn_sc_relu(input, 66, kernel=(3, 3), stride=2, padding='same', drop_rate=params.drop_rate)
        conv2 = self.residual(conv1, 128, kernel=(3, 3), stride=1, padding='same', _shortcut=True)
        conv3 = self.conv_bn_sc_relu(conv2, 128, kernel=(3, 3), stride=1, padding='same', drop_rate=params.drop_rate)
        down1 = self.residual(conv3, 192, kernel=(3, 3), stride=(2,2), padding='same', _shortcut=True)
        conv4 = self.residual(down1, 192, kernel=(3, 3), stride=1, padding='same', _shortcut=True)
        conv5 = self.conv_bn_sc_relu(conv4, 192, kernel=(3, 3), stride=1, padding='same', drop_rate=params.drop_rate)
        conv6 = self.conv_bn_sc_relu(conv5, 128, kernel=(3, 3), stride=1, padding='same', drop_rate=params.drop_rate)
        down2 = self.residual(conv6, 288, kernel=(3, 3), stride=(2,2), padding='same', _shortcut=True)
        conv11 = self.residual(down2, 288, kernel=(3, 3), stride=1, padding='same', _shortcut=True)
        conv12 = self.conv_bn_sc_relu(conv11, 355, kernel=(3, 3), stride=1, padding='same', drop_rate=params.drop_rate)
        conv13 = self.conv_bn_sc_relu(conv12, 432, kernel=(3, 3), stride=1, padding='same', drop_rate=params.drop_rate)
        gmpool = self.saf_pool(conv13, 2, stride=2, padding='valid')
        flatten = kl.Flatten()(gmpool)
        logits = kl.Dense(self.classes)(flatten)

        self.ios = [input, logits]
        self.model = keras.models.Model(input, logits, name='SimpleNet_v2')
        self.model.summary()

    def loss(self):
        def loss_(y_true, y_pred):
            return keras.losses.categorical_crossentropy(y_true, y_pred, True)
        return loss_
