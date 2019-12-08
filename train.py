from dataset import DatasetGenerator
from networks import callbacks
from networks import Network
import os
import numpy as np
import tensorflow as tf


def named_logs(names, logs):
    result = {}
    for l in zip(names, logs):
        result[l[0]] = l[1]
    return result


def train(params, net:Network, model:tf.keras.models.Model, model_path:str, train_dataset:DatasetGenerator, validation_dataset:DatasetGenerator):

    sgd_optimizer = tf.keras.optimizers.SGD(lr=params.learning_rate)
    model.compile(optimizer=sgd_optimizer, loss=net.loss(), metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.CategoricalCrossentropy()])

    # Start training
    if params.mode == 'train':
        # set log names
        train_names = ['train_loss', 'train_acc', 'train_categorical_crossentropy']
        val_names = ['val_loss', 'val_acc', 'val_categorical_crossentropy']
        avg_val_names = ['avg_val_loss', 'avg_val_acc', 'val_categorical_crossentropy']
        # set callbacks
        log_dir = os.path.join(params.checkpoints_dir, params.name + '_v' + str(params.version) + '/')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              write_graph=True,
                                                              write_grads=True)
        tensorboard_callback.set_model(model)
        es = callbacks.EarlyStopping(patience=5,mode=['min','max','min'])

        for epoch in range(params.start_epoch, params.epochs):
            es.on_epoch_begin(epoch)
            print('Epoch {:d}/{:d}'.format(epoch+1,params.epochs+1))
            for step in range(len(train_dataset)):
                x, y = train_dataset.__getitem__(step)
                if step % params.log_fq == 0 or step + 1 == len(train_dataset):
                    print('{:d}/{:d}'.format(step,len(train_dataset)-1))
                    loss, accuracy, cross_ent = model.train_on_batch(x,y)
                    print(' - loss: {:.4f} - acc: {:.4f} - cross_ent: {:.4f}'.format(loss,accuracy,cross_ent))
                    tensorboard_callback.on_epoch_end(len(train_dataset) * epoch + step, named_logs(train_names, [loss, accuracy, cross_ent]))

            model.save_weights(model_path)
            print('Weights saved to', model_path)

            print('Evaluation')
            avg_loss = 0; avg_acc = 0; avg_ce = 0
            for step in range(len(validation_dataset)):
                x, y = validation_dataset.__getitem__(step)
                loss, accuracy, cross_ent = model.test_on_batch(x,y)
                es.on_batch_end(step, named_logs(val_names, [loss, accuracy, cross_ent]))
                if step % params.log_fq == 0 or step + 1 == len(validation_dataset):
                    print('{:d}/{:d} - loss: {:.4f} - acc: {:.4f} - cross_ent: {:.4f}'.format(step, len(validation_dataset)-1, loss, accuracy, cross_ent))
                    tensorboard_callback.on_epoch_end(len(validation_dataset) * epoch + step, named_logs(val_names, [loss, accuracy, cross_ent]))
                avg_loss += loss
                avg_acc += accuracy
                avg_ce += cross_ent
            avg_loss /= len(validation_dataset)
            avg_acc /= len(validation_dataset)
            avg_ce /= len(validation_dataset)
            tensorboard_callback.on_epoch_end(epoch, named_logs(avg_val_names, [avg_loss, avg_acc, avg_ce]))
            es.on_epoch_end(epoch)
            # On Epoch End
            print('\n')
            train_dataset.on_epoch_end()
            validation_dataset.on_epoch_end()
            if es.early_stop:
                print('Early stopping due to non improving any of given metrics.')
                es.print_logs()
                es_model_path = os.path.join(params.checkpoints_dir,
                                    params.name + '/{:s}_v{:d}_es_loss{:.5f}acc{:.5f}.h5'.format(params.name, params.version, avg_loss, avg_acc))
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

