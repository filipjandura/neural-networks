from dataset.cifar import DatasetGenerator

import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from scipy.special import softmax
from sklearn import metrics
from matplotlib import pyplot as plt
from skimage.color import hsv2rgb

np.random.seed(2019)


def tpr_fpr(y_pred, y_true, threshold):
    '''
    Computes true positive rate, false positive rate and precision
    :param result Zipped [y_pred, y_true]
    :param threshold int array of threshold values (0,1)
    '''
    print("tpr_fpr",threshold)
    tp = sum(1 for p,t in zip(y_pred,y_true) if p >= threshold and t == 1)
    fp = sum(1 for p,t in zip(y_pred,y_true) if p >= threshold and t == 0)
    tn = sum(1 for p,t in zip(y_pred,y_true) if p < threshold and t == 0)
    fn = sum(1 for p,t in zip(y_pred,y_true) if p < threshold and t == 1)
    pcond = tp + fp
    if pcond > 0:
        precision = tp/pcond
    else:
        precision = 0
    tcond = tp + fn
    if tcond > 0:
        tpr = tp / tcond
    else:
        tpr = 0
    fcond = tn + fp
    if fcond > 0:
        fpr = fp / fcond
    else:
        fpr = 0
    return tpr, fpr, precision

def roc_draw(y_pred, y_true, thresholds):
    tpr, fpr, precision = np.array([tpr_fpr(y_pred, y_true, th) for th in thresholds]).transpose()
    auc = metrics.auc(fpr, tpr)
    figure = plt.figure(figsize=[12,6])
    figure.add_subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='red', label='ROC auc: {:0.4f}'.format(auc))
    plt.plot([1, 0], [1, 0], color='navy', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    figure.add_subplot(1, 2, 2)
    plt.plot(tpr, precision, color='darkorange', label='PR curve')
    plt.plot([1,0],[1,0], color='navy', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")

    return figure

def plot_image_to_summary(image:np.ndarray, title:str):
    fig = plt.figure()
    plt.imshow(image)
    plt.title(title)
    return plot_figure_to_summary(figure=fig)

def plot_figure_to_summary(figure:plt.Figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  plt.close(figure)
  buf.seek(0)
  image_string = buf.getvalue()
  buf.close()
  return tf.compat.v1.Summary.Image(height=int(figure.get_figheight())*100,
                          width=int(figure.get_figwidth())*100,
                          colorspace=4,
                          encoded_image_string=image_string)
  # # Convert PNG buffer to TF image
  # image = tf.image.decode_png(buf.getvalue(), channels=4)
  # # Add the batch dimension
  # image = tf.expand_dims(image, 0)
  # return image


def write_image(file_writer, image_logs:dict):
    for name, image in image_logs.items():
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=name, image=image)])
        file_writer.add_summary(summary)


def write_logs(file_writer, logs):
    for name, value in logs.items():
        if isinstance(value, np.ndarray):
            value = value.item()
        summary = tf.compat.v1.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        file_writer.add_summary(summary)
    file_writer.flush()


def named_logs(names, logs):
    result = {}
    for l in zip(names, logs):
        result[l[0]] = l[1]
    return result


def test(params, net, model:tf.keras.models.Model, test_dataset:DatasetGenerator):

    test_dataset.mode = params.mode
    sgd_optimizer = tf.keras.optimizers.SGD(lr=params.learning_rate)
    model.compile(optimizer=sgd_optimizer, loss=net.loss(), metrics=[tf.keras.metrics.CategoricalAccuracy(name='top_1_categorical_accuracy'),
                                                                     tf.keras.metrics.TopKCategoricalAccuracy(2,name='top_2_categorical_accuracy'),
                                                                     tf.keras.metrics.TopKCategoricalAccuracy(3,name='top_3_categorical_accuracy'),
                                                                     tf.keras.metrics.TopKCategoricalAccuracy(5,name='top_5_categorical_accuracy')])
    # Start testing
    if params.mode == 'test':
        # set log names
        test_names = model.metrics_names[1:]

        log_dir = os.path.join(params.checkpoints_dir, params.name + '_v' + str(params.version) + '/test/')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_writer = tf.summary.FileWriter(log_dir, K.get_graph(), flush_secs=60)

        print('Testing')
        accuracy_1, accuracy_2, accuracy_3, accuracy_5 = [0,0,0,0]
        steps = test_dataset.test_y.shape[0] // params.batch_size
        for step in range(steps):
            x, y = test_dataset.__getitem__(step)

            _, accuracy_1, accuracy_2, accuracy_3, accuracy_5 = model.test_on_batch(x, y, reset_metrics=False)

            if step % params.log_fq == 0 or step + 1 == steps:
                print('{:d}/{:d}'.format(step, steps-1))
                print('   Accuracy top_1: {:.4f}'.format(accuracy_1))
                print('   Accuracy top_2: {:.4f}'.format(accuracy_2))
                print('   Accuracy top_3: {:.4f}'.format(accuracy_3))
                print('   Accuracy top_5: {:.4f}'.format(accuracy_5))

        # On Epoch End
        print('\n')
        logs = named_logs(test_names, [accuracy_1, accuracy_2, accuracy_3, accuracy_5])
        write_logs(file_writer, logs)

        print('Model Confidence testing ROC + PRc - evaluating predictions')
        thresholds = np.arange(10)/10
        thresholds[0] = 1.e-7
        thresholds[-1] = 1-1.e-7
        pred_total = []
        y_total = []

        classes = np.concatenate((np.arange(params.num_classes, dtype=int).reshape((-1,1)), np.zeros((params.num_classes,1))), axis=1)
        saved_wrongs = []

        for step in range(steps):
            x, y = test_dataset.__getitem__(step)
            y_pred = model.predict_on_batch(x)
            s_max = softmax(y_pred, axis=1)
            y_am = np.argmax(y, axis=1)
            y_p_am = np.argmax(y_pred, axis=1)
            wrong = np.array([[y_am[i], y_p_am[i], x[i,:,:,3:]] for i in range(len(y_am)) if y_am[i]!=y_p_am[i] and y_am[i] in classes[:,0]])
            for i in classes:
                there = np.where(wrong[:,0]==i[0])[0]
                if len(there) > 0:
                    for t in there:
                        if i[1] < 5:
                            saved_wrongs.append([i[0], wrong[t,1], hsv2rgb(wrong[t,2])])
                            i[1] += 1
                        else:
                            break
            pred_total += list(s_max.flatten())
            y_total += list(y.flatten())
            if step % params.log_fq == 0 or step + 1 == steps:
                print('{:d}/{:d}'.format(step, steps-1))
                for i in range(20):
                    print('[{:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f}]'.format(y_pred[i][0], y_pred[i][1], y_pred[i][2], y_pred[i][3], y_pred[i][4], y_pred[i][5], y_pred[i][6], y_pred[i][7], y_pred[i][8], y_pred[i][9]))
                    print('[{:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f}]'.format(s_max[i][0], s_max[i][1], s_max[i][2], s_max[i][3], s_max[i][4], s_max[i][5], s_max[i][6], s_max[i][7], s_max[i][8], s_max[i][9]))
                    print('[{:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f}]'.format(y[i][0], y[i][1], y[i][2], y[i][3], y[i][4], y[i][5], y[i][6], y[i][7], y[i][8], y[i][9]))
                    print('\n')
        figure = roc_draw(pred_total, y_total, thresholds=thresholds)
        image_summary = plot_figure_to_summary(figure)
        write_image(file_writer, {'ROC_PR':image_summary})

        for save in saved_wrongs:
            image_summary = plot_image_to_summary(save[2], str(int(save[0])) + ' | ' + str(save[1]))
            write_image(file_writer, {'True_Class_'+str(int(save[0])): image_summary})
            write_image(file_writer, {'False_Class_'+str(int(save[1])): image_summary})
        file_writer.close()

        return 0

