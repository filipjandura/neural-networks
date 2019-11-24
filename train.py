from dataset.cifar import DatasetCIFAR10 as CIFAR10
from config import Options
from networks import SimpleNet
from networks import callbacks
from keras.utils import plot_model

import os
import numpy as np
import tensorflow as tf

np.random.seed(2019)

def get_network(params):
    if params.name == 'simplenet':
        simple_net = SimpleNet(params.num_classes, params.batch_size, params.img_size, params.nc)
        simple_net.build_network(params)
        return simple_net, simple_net.model
    elif params.name == 'simplenet_residual':
        pass

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names,logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def named_logs(names, logs):
    result = {}
    for l in zip(names, logs):
        result[l[0]] = l[1]
    return result

def main():
    options = Options()
    params = options.parse()

    train_dataset = CIFAR10(params.data_root, params.batch_size, params.img_size, params.nc, True)
    # update number of channels after augmentation
    params.nc = train_dataset.channels

    count = int(train_dataset.train_idx.shape[0] * 0.9)
    train_idx_range = np.random.choice(train_dataset.train_idx, count, replace=False)
    val_idx_range = np.delete(train_dataset.train_idx, train_idx_range)

    train_dataset.train_idx = train_idx_range
    validation_dataset = CIFAR10(params.data_root, params.batch_size, params.img_size, params.nc, True)
    validation_dataset.train_idx = val_idx_range

    model_path = os.path.join(params.checkpoints_dir, params.name + '/' + params.name + '.h5')

    net, model = get_network(params)

    if os.path.exists(model_path):
        try:
            model.load_weights(model_path)
            print('Weights loaded from:')
            print(model_path)

        except ValueError as e:
            print('{0}'.format(e))
            print('Wrong format of saved weights. Use different model and saved configurations')

    sgd_optimizer = tf.keras.optimizers.SGD(lr=params.learning_rate)
    model.compile(optimizer=sgd_optimizer, loss=net.loss(), metrics=['accuracy'])

    plot_model(model,
               os.path.join(params.checkpoints_dir, params.name+'/model.png'),
               show_shapes=True, show_layer_names=True)

    # Start training
    if params.mode == 'train':
        # set log names
        train_names = ['train_loss', 'train_acc']
        val_names = ['val_loss', 'val_acc']
        avg_val_names = ['avg_val_loss', 'avg_val_acc']
        # set callbacks
        log_dir = os.path.join(params.checkpoints_dir, params.name + '/')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              write_graph=True,
                                                              write_grads=True)
        tensorboard_callback.set_model(model)
        es = callbacks.EarlyStopping(patience=10,mode=['min','max'])

        for epoch in range(params.start_epoch, params.epochs):
            es.on_epoch_begin(epoch)
            for step in range(len(train_dataset)):
                x, y = train_dataset.__getitem__(step)
                print('{:d}/{:d}'.format(step,len(train_dataset)))
                loss, accuracy = model.train_on_batch(x,y)
                print(' - loss: {:.4f} - acc: {:.4f}'.format(loss,accuracy))
                if step % params.log_fq == 0 or step + 1 == len(train_dataset):
                    tensorboard_callback.on_epoch_end(len(train_dataset) * epoch + step, named_logs(train_names, [loss, accuracy]))

            model.save_weights(model_path)
            print('Weights saved to', model_path)

            print('Evaluation')
            avg_loss = 0; avg_acc = 0
            for step in range(len(validation_dataset)):
                x, y = validation_dataset.__getitem__(step)
                loss, accuracy = model.test_on_batch(x,y)
                print('{:d}/{:d} - loss: {:.4f} - acc: {:.4f}'.format(step, len(validation_dataset), loss, accuracy))
                es.on_batch_end(step, named_logs(val_names, [loss, accuracy]))
                if step % params.log_fq == 0 or step + 1 == len(validation_dataset):
                    tensorboard_callback.on_epoch_end(len(validation_dataset) * epoch + step, named_logs(val_names, [loss, accuracy]))
                avg_loss += loss
                avg_acc += accuracy
            avg_loss /= len(validation_dataset)
            avg_acc /= len(validation_dataset)
            tensorboard_callback.on_epoch_end(epoch, named_logs(avg_val_names, [avg_loss, avg_acc]))
            es.on_epoch_end(epoch)
            # On Epoch End
            print('\n')
            train_dataset.on_epoch_end()
            validation_dataset.on_epoch_end()
            if es.early_stop:
                print('Early stopping due to non improving any of given metrics.')
                es.print_logs()
                es_model_path = os.path.join(params.checkpoints_dir,
                                    params.name + '/{:s}_es_loss{:.5f}acc{:.5f}.h5'.format(params.name, avg_loss, avg_acc))
                model.save_weights(es_model_path)
                print('Weights of early stopped model saved to', es_model_path)
                break

        es.on_train_end()
        tensorboard_callback.on_train_end(None)

        # if you want to constrain training to certain number of iterations per epoch
        # steps_per_epoch=params.steps_per_epoch
        # model.fit_generator(generator=train_dataset, validation_data=validation_dataset,
        #                     epochs=params.epochs,
        #                     use_multiprocessing=False,
        #                     workers=1)

if __name__ == "__main__":
    main()
