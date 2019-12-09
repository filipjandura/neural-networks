from dataset.dataset import DatasetGenerator
from dataset.util import zca_whitening, map_rgb2hsv
import numpy as np
import os
import pickle


class DatasetCIFAR100(DatasetGenerator):
    def __init__(self, path, batch_size, image_size=32, channels=3, normalize=True):
        print('Loading Dataset CIFAR-100 from ' + path)
        DatasetGenerator.__init__(self, path, batch_size, image_size, channels, normalize)

        self.classes = 100
        self.train_data_length = 0
        self.train_files = []
        self.load_dataset()
        self.train_original = self.train_x
        self.test_original  = self.test_x
        if normalize:
            if self.preprocessed == False:
                self._normalize()
                self._augment(True)
                self.__save_dataset()
                self.preprocessed = True
                self._reset()
                self.load_dataset()
        self.last_train_file = None
        self.dimensions = [32, 32, 3]
        self.mode = 'train'

        self.validation_dataset = None

        print('Loaded CIFAR-100 from: ', path)

    def set_validation_dataset(self, dataset:DatasetGenerator):
        self.validation_dataset = dataset
        self.validation_dataset.train_files = [self.train_files[-1]]
        self.validation_dataset.train_idx = np.arange(self.train_files[-1][1])
        self.validation_dataset.train_data_length = self.validation_dataset.train_idx.shape[0]
        self.train_files.remove(self.validation_dataset.train_files[0])
        return self.validation_dataset

    def _reset(self):
        DatasetGenerator._reset(self)
        self.train_cy = []
        self.test_cy = []
        self.test_original = []
        self.train_original = []

    def __getitem__(self, item):
        # print(item)
        count = 0
        index = item
        for i in range(len(self.train_files)):
            count += self.train_files[i][1]
            if item < count:
                if self.train_files[i] != self.last_train_file:
                    self.train_x, self.train_y, self.train_cy = self.__unpickle(self.train_files[i][0])
                    self.train_idx = np.arange(self.train_y.shape[0])
                    np.random.shuffle(self.train_idx)
                if i > 0:
                    index = item - (count - self.train_files[i][1])
                self.last_train_file = self.train_files[i]
                break

        if self.mode == 'train':
            start_idx = index * self.batch_size
            end_idx = (index + 1) * self.batch_size

            if (index + 1) > self.last_train_file[1]:
                start_idx = self.train_idx.shape[0] - self.batch_size
                end_idx = self.train_idx.shape[0]

            # print("end - start ", end_idx - start_idx)
            batch_x = self.train_x[self.train_idx[start_idx:end_idx]]
            batch_y = self.train_y[self.train_idx[start_idx:end_idx]]

            return (batch_x, batch_y)

        if self.mode == 'test':
            start_idx = item * self.batch_size
            end_idx = (item + 1) * self.batch_size

            if (item+1) > self.__len__():
                start_idx = self.test_idx.shape[0] - self.batch_size
                end_idx = self.test_idx.shape[0]

            # print("end - start ", end_idx - start_idx)
            batch_x = self.test_x[self.test_idx[start_idx:end_idx]]
            batch_y = self.test_y[self.test_idx[start_idx:end_idx]]

            return (batch_x, batch_y)

    def __len__(self):
        return self.train_data_length // self.batch_size

    def __num_classes__(self):
        return self.classes

    def on_epoch_end(self):
        np.random.shuffle(self.train_files)

    def load_dataset(self):
        if len(self.train_x) > 0 and len(self.test_x) > 0:
            return

        files = os.listdir(self.path)
        files.sort()
        preprocessed = False
        for file in files:
            if 'preprocessed' in file:
                preprocessed = True
                break

        if preprocessed:
            for f in files:
                file = os.path.join(self.path, f)
                if 'preprocessed_train' in file:
                    data, _, _ = self.__unpickle(file)
                    self.train_data_length += data.shape[0]
                    self.train_files.append([file, data.shape[0]])
                    self.channels = data.shape[3]
                if 'preprocessed_test' in file:
                    self.test_x, self.test_y, self.test_cy = self.__unpickle(file)
            self.preprocessed = True

            print('Loaded {:d} train images and {:d} test images'.format(self.train_data_length, self.test_y.shape[0]))

        else:
            if len(files) > 0:
                for f in files:
                    file = os.path.join(self.path, f)
                    data, labels, coarse_labels = self.__unpickle(file)
                    if 'train' in file.lower():
                        if len(self.train_x) == 0:
                            self.train_x = np.array(data, dtype=np.float32)
                            self.train_y = np.array(labels)
                            self.train_cy = np.array(coarse_labels)
                        else:
                            self.train_x = np.concatenate((self.train_x, np.array(data)), axis=0)
                            self.train_y = np.concatenate((self.train_y, np.array(labels)), axis=0)
                            self.train_cy = np.concatenate((self.train_cy, np.array(coarse_labels)), axis=0)
                    else:
                        self.test_x = np.array(data, dtype=np.float32)
                        self.test_y = np.array(labels)
                        self.test_cy = np.array(coarse_labels)

            one_hot_labels = np.zeros((len(self.train_y), self.classes), dtype=np.float32)
            one_hot_labels[np.arange(len(self.train_y)), self.train_y] = 1
            self.train_y = one_hot_labels

            one_hot_coarse_labels = np.zeros((len(self.train_cy), np.max(self.train_cy) + 1), dtype=np.float32)
            one_hot_coarse_labels[np.arange(len(self.train_cy)), self.train_cy] = 1
            self.train_cy = one_hot_coarse_labels

            one_hot_labels = np.zeros((len(self.test_y), self.classes), dtype=np.float32)
            one_hot_labels[np.arange(len(self.test_y)), self.test_y] = 1
            self.test_y = one_hot_labels

            self.train_x = self.train_x.reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1]) / 255
            self.test_x = self.test_x.reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1]) / 255

    def _augment(self, more=True):
        print('Adding HSV color space coding.')
        self.train_x = np.concatenate((self.train_x, map_rgb2hsv(self.train_original)), axis=-1)
        self.test_x  = np.concatenate((self.test_x, map_rgb2hsv(self.test_original)), axis=-1)

        print('Adding mirrored images by X axis')
        # (np.fliplr) => np.flip(axis=1) is equivalent to self.train_x[...,::-1,...]
        self.train_x = np.concatenate((self.train_x, np.flip(self.train_x, axis=2)),axis=0)
        self.train_y = np.concatenate((self.train_y, self.train_y), axis=0)
        self.train_cy = np.concatenate((self.train_cy, self.train_cy), axis=0)

        if more:
            print('Adding mirrored images by Y axis')
            self.train_x = np.concatenate((self.train_x, np.flip(self.train_x, axis=1)), axis=0)
            self.train_y = np.concatenate((self.train_y, self.train_y), axis=0)
            self.train_cy = np.concatenate((self.train_cy, self.train_cy), axis=0)
            print('Adding images rotated by 90 degrees') # 180 and 270 are redundant - draw it out to see why
            self.train_x = np.concatenate((self.train_x, np.rot90(self.train_x, k=1, axes=(1,2))), axis=0)
            self.train_y = np.concatenate((self.train_y, self.train_y), axis=0)
            self.train_cy = np.concatenate((self.train_cy, self.train_cy), axis=0)

        self.channels = self.train_x.shape[3]
        print('Images now have', self.channels, 'channels - R,G,B,H,S,V.')

    def _normalize(self, batch=5000, scale_mode='item'):
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
        with open(os.path.join(self.path, 'preprocessed_test'), mode='wb') as file:
            data_test = {b'data': self.test_x, b'fine_labels': self.test_y, b'coarse_labels': self.test_cy}
            pickle.dump(data_test, file, pickle.HIGHEST_PROTOCOL)
        print('Data saved to: ')
        print(os.path.join(self.path, 'preprocessed_test'))
        idxs = np.arange(self.train_y.shape[0])
        np.random.shuffle(idxs)
        self.train_x = self.train_x[idxs]
        self.train_y = self.train_y[idxs]
        self.train_cy = self.train_cy[idxs]

        size = len(self.train_x) // 10
        for i in range(10):
            data_train = {b'data': self.train_x[i * size:(i + 1) * size],
                          b'fine_labels': self.train_y[i * size:(i + 1) * size],
                          b'coarse_labels': self.train_cy[i * size:(i + 1) * size]}
            with open(os.path.join(self.path, 'preprocessed_train_' + str(i)), mode='wb') as file:
                pickle.dump(data_train, file, pickle.HIGHEST_PROTOCOL)
                print('Data saved to: ')
                print(file.name)

    def __unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        if self.classes == 10:
            return dict[b'data'], dict[b'labels']
        elif self.classes == 100:
            return dict[b'data'], dict[b'fine_labels'], dict[b'coarse_labels']


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
        self.mode = 'train'

        print('Loaded CIFAR-10 from: ', path)

    def __getitem__(self, item):
        # print(item)
        if self.mode == 'train':
            start_idx = item * self.batch_size
            end_idx = (item + 1) * self.batch_size

            if (item + 1) > self.__len__():
                start_idx = self.train_idx.shape[0] - self.batch_size
                end_idx = self.train_idx.shape[0]

            # print("end - start ", end_idx - start_idx)
            batch_x = self.train_x[self.train_idx[start_idx:end_idx]]
            batch_y = self.train_y[self.train_idx[start_idx:end_idx]]

            return (batch_x, batch_y)

        if self.mode == 'test':
            start_idx = item * self.batch_size
            end_idx = (item + 1) * self.batch_size

            if (item+1) > self.__len__():
                start_idx = self.test_idx.shape[0] - self.batch_size
                end_idx = self.test_idx.shape[0]

            # print("end - start ", end_idx - start_idx)
            batch_x = self.test_x[self.test_idx[start_idx:end_idx]]
            batch_y = self.test_y[self.test_idx[start_idx:end_idx]]

            return (batch_x, batch_y)

    def __num_classes__(self):
        return self.classes

    def load_dataset(self):
        if len(self.train_x) > 0 and len(self.test_x) > 0:
            return

        files = os.listdir(self.path)
        processed = False
        for file in files:
            if 'processed' in file:
                processed = True
                break

        if processed:
            for f in files:
                file = os.path.join(self.path, f)
                if 'preprocessed_train' in file:
                    self.train_x, self.train_y = self.__unpickle(file)
                if 'preprocessed_test' in file:
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
        data_train = {b'data': self.train_x, b'labels': self.train_y}
        data_test = {b'data': self.test_x, b'labels': self.test_y}
        with open(os.path.join(self.path, 'preprocessed_train'), mode='wb') as file:
            pickle.dump(data_train, file, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.path, 'preprocessed_test'), mode='wb') as file:
            pickle.dump(data_test, file, pickle.HIGHEST_PROTOCOL)
        print('Data saved to: ',
              os.path.join(self.path, 'preprocessed_train'),
              os.path.join(self.path, 'preprocessed_test'))
        return

    def __unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            return dict[b'data'], dict[b'labels']
