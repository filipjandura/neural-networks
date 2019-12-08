from config import Options
from train import train
from test import test
from networks import SimpleNet
from networks import SimpleNetResidual
from dataset import DatasetCIFAR10
from dataset import DatasetCIFAR100
import tensorflow as tf

import os
import numpy as np

def get_network(params):
    if params.name == 'simplenet':
        simple_net = SimpleNet(params.num_classes, params.batch_size, params.img_size, params.nc)
        simple_net.build_network(params)
        return simple_net, simple_net.model
    elif params.name == 'simplenet_residual':
        residual_simple_net = SimpleNetResidual(params.num_classes, params.batch_size, params.img_size, params.nc)
        residual_simple_net.build_network(params)
        return residual_simple_net, residual_simple_net.model

def get_dataset(params):
    if params.num_classes == 10:
        train_dataset = DatasetCIFAR10(params.data_root, params.batch_size, params.img_size, params.nc, True)
        validation_dataset = DatasetCIFAR10(params.data_root, params.batch_size, params.img_size, params.nc, True)

        count = int(train_dataset.train_idx.shape[0] * 0.9)
        train_idx_range = np.random.choice(train_dataset.train_idx, count, replace=False)
        val_idx_range = np.delete(train_dataset.train_idx, train_idx_range)
        train_dataset.keep_indices(train_idx_range)
        validation_dataset.keep_indices(val_idx_range)

    elif params.num_classes == 100:
        train_dataset = DatasetCIFAR100(params.data_root, params.batch_size, params.img_size, params.nc, True)
        validation_dataset = DatasetCIFAR100(params.data_root, params.batch_size, params.img_size, params.nc, True)
        validation_dataset = train_dataset.set_validation_dataset(validation_dataset)
    else:
        return None, None

    # update number of channels after augmentation
    params.nc = train_dataset.channels
    train_dataset.mode = params.mode
    return train_dataset, validation_dataset

if __name__ == "__main__":
    options = Options()
    params = options.parse()

    train_dataset, validation_dataset = get_dataset(params)

    net, model = get_network(params)
    tf.keras.utils.plot_model(model,
               os.path.join(params.checkpoints_dir, params.name + '_v' + str(params.version) +'/model_v'+str(params.version)+'.png'),
               show_shapes=True, show_layer_names=True)

    model_path = os.path.join(params.checkpoints_dir, params.name + '_v' + str(params.version) + '/' + params.name + '_v' + str(params.version) + '.h5')
    if os.path.exists(model_path):
        try:
            model.load_weights(model_path)
            print('Weights loaded from:')
            print(model_path)

        except ValueError as e:
            print('{0}'.format(e))
            print('Wrong format of saved weights. Use different model and saved configurations')

    if params.mode == 'train':
        train(params, net, model, model_path, train_dataset, validation_dataset)

    if params.mode == 'test':
        test(params, net, model, test_dataset=train_dataset)
