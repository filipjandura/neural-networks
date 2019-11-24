from dataset.dataset import DatasetGenerator
from dataset.util import zca_whitening, map_rgb2hsv
import numpy as np
import os
import pickle


class DatasetCIFAR10(DatasetGenerator):
    def __init__(self, path, batch_size, image_size=32, channels=3, normalize=True):
        print('Loading Dataset CIFAR-10 from ' + path)
        DatasetGenerator.__init__(self, path, batch_size, image_size, channels, normalize)

        self.classes = 10
        self.load_dataset()
        self.train_original = self.train_x
        self.test_original  = self.test_x
        if normalize:
            if self.preprocessed == False:
                self._normalize()
                self._augment()
                self.__save_dataset()
                self.preprocessed = True

        self.train_idx = np.arange(self.train_y.shape[0])
        self.test_idx = np.arange(self.test_y.shape[0])

        self.dimensions = [32, 32, 3]
        self.train_data_length = self.train_x.shape[0]

        print("Loaded CIFAR-10 from: ", path)

    def __getitem__(self, item):
        # print(item)
        start_idx = item * self.batch_size
        end_idx = (item + 1) * self.batch_size

        if (item+1) > self.__len__():
            start_idx = self.train_idx.shape[0] - self.batch_size
            end_idx = self.train_idx.shape[0]

        # print("end - start ", end_idx - start_idx)
        batch_x = self.train_x[self.train_idx[start_idx:end_idx]]
        batch_y = self.train_y[self.train_idx[start_idx:end_idx]]

        return (batch_x, batch_y)

    def __num_classes__(self):
        return self.classes

    def load_dataset(self):
        if len(self.train_x) > 0 and len(self.test_x) > 0:
            return

        files = os.listdir(self.path)

        if 'preprocessed_train' in files:
            file = os.path.join(self.path, 'preprocessed_train')
            self.train_x, self.train_y = self.__unpickle(file)

            file = os.path.join(self.path, 'preprocessed_test')
            self.test_x, self.test_y = self.__unpickle(file)
            self.preprocessed = True
            print('Loaded {:d} train images and {:d} test images'.format(self.train_y.shape[0], self.test_y.shape[0]))
            self.channels = self.train_x.shape[3]
        else:
            if len(files) > 0:
                for f in files:
                    file = os.path.join(self.path, f)
                    data, labels = self.__unpickle(file)
                    if not 'test' in file.lower():
                        if len(self.train_x) == 0:
                            self.train_x = np.array(data, dtype=np.float32)
                            self.train_y = np.array(labels)
                        else:
                            self.train_x = np.concatenate((self.train_x, np.array(data)), axis=0)
                            self.train_y = np.concatenate((self.train_y, np.array(labels)), axis=0)
                    else:
                        self.test_x = np.array(data, dtype=np.float32)
                        self.test_y = np.array(labels)

            one_hot_labels = np.zeros((len(self.train_y), self.classes), dtype=np.float32)
            one_hot_labels[np.arange(len(self.train_y)), self.train_y] = 1
            self.train_y = one_hot_labels

            one_hot_labels = np.zeros((len(self.test_y), self.classes), dtype=np.float32)
            one_hot_labels[np.arange(len(self.test_y)), self.test_y] = 1
            self.test_y = one_hot_labels

            self.train_x = self.train_x.reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1]) / 255
            self.test_x = self.test_x.reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1]) / 255

    def _augment(self):
        print('Adding HSV color space coding.')
        self.train_x = np.concatenate((self.train_x, map_rgb2hsv(self.train_original)), axis=-1)
        self.test_x  = np.concatenate((self.test_x, map_rgb2hsv(self.test_original)), axis=-1)

        print('Adding mirrored images by X axis')
        # (np.fliplr) => np.flip(axis=1) is equivalent to self.train_x[...,::-1,...]
        self.train_x = np.concatenate((self.train_x, np.flip(self.train_x, axis=2)),axis=0)
        self.train_y = np.concatenate((self.train_y, self.train_y), axis=0)

        self.channels = self.train_x.shape[3]
        print('Images now have', self.channels, 'channels - R,G,B,H,S,V.')

    def _normalize(self, batch=2000, scale_mode='item'):
        ## Apply zca whitening and contrast normalization
        through_data = int(self.train_x.shape[0] / batch)
        for i in range(through_data):
            print("Normalizing train {:d}-{:d}/{:d}".format(i * batch, (i + 1) * batch, self.train_x.shape[0]))
            data = self.train_x[i * batch: (i + 1) * batch]
            self.train_x[i * batch: (i + 1) * batch] = zca_whitening(data)
        through_data = int(self.test_x.shape[0] / batch)
        for i in range(through_data):
            print("Normalizing test {:d}-{:d}/{:d}".format(i * batch, (i + 1) * batch, self.test_x.shape[0]))
            data = self.test_x[i * batch: (i + 1) * batch]
            self.test_x[i * batch: (i + 1) * batch] = zca_whitening(data)
        print("Data normalized by ZCA whitening")

        ## Apply Pixel standardization - scaling pixel values to have a zero mean and unit variance
        if scale_mode == 'item':
            zmuv_n = lambda x: (x - x.mean()) / x.std()
            self.train_x = np.array(list(map(zmuv_n, self.train_x)))
            self.test_x = np.array(list(map(zmuv_n, self.test_x)))
        else:
            self.train_x = (self.train_x - self.train_x.mean()) / self.train_x.std()
            self.test_x = (self.test_x - self.test_x.mean()) / self.test_x.std()
        print("Data scaled to Zero Mean Unit Variance")

    def __save_dataset(self):
        data_train = {b'data':self.train_x, b'labels':self.train_y}
        data_test = {b'data':self.test_x, b'labels':self.test_y}

        with open(os.path.join(self.path,'preprocessed_train'),mode='wb') as file:
            pickle.dump(data_train, file, pickle.DEFAULT_PROTOCOL)
        with open(os.path.join(self.path,'preprocessed_test'), mode='wb') as file:
            pickle.dump(data_test, file, pickle.DEFAULT_PROTOCOL)
        print('Data saved to: ',
              os.path.join(self.path,'preprocessed_train'),
              os.path.join(self.path,'preprocessed_test'))

    def __unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict[b'data'], dict[b'labels']
