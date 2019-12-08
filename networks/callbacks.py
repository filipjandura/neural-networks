import numpy as np
from tensorflow import keras

class EarlyStopping(keras.callbacks.Callback):
    def __init__(self, patience=0, mode='min'):
        keras.callbacks.Callback.__init__(self)
        self.batch_no = 0
        self.patience = patience
        self.mode = mode

        self.epoch = 0
        self.last_logs = []
        self.early_stop = False

    def on_batch_end(self, batch, logs=None):
        self.batch_no = batch

        for k in logs.keys():
            if self.last_logs[len(self.last_logs)-1].get(k) == None:
                self.last_logs[len(self.last_logs)-1][k] = logs[k]
            else:
                self.last_logs[len(self.last_logs) - 1][k] += logs[k]

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if self.epoch > self.patience:
            self.last_logs[:-1] = self.last_logs[1:]
            self.last_logs[-1] = {}
        else:
            self.last_logs.append({})

    def on_epoch_end(self, epoch, logs=None):
        for k in self.last_logs[-1].keys():
            self.last_logs[-1][k]=self.last_logs[-1][k] / self.batch_no
        results = []
        if self.patience < epoch:
            m = 0
            for k in self.last_logs[-1].keys():
                mode = self.mode[m]
                m += 1
                if mode == 'min':
                    for log in self.last_logs:
                        results.append(self.last_logs[-1][k] < log[k])
                elif mode == 'max':
                    for log in self.last_logs:
                        results.append(self.last_logs[-1][k] > log[k])
                else:
                    break
            self.early_stop = True not in results # Master Yoda style
        else:
            self.early_stop = False

    def print_logs(self):
        i = len(self.last_logs)
        for logs in self.last_logs:
            values = '. '
            for k in logs.keys():
                values += k + ': ' + str(logs[k]) + '; '

            print('Epoch ', self.epoch - i, values)
            i -= 1
