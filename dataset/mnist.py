from dataset.dataset import DatasetGenerator
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class DatasetMNIST(DatasetGenerator):
    def __init__(self, path, batch_size, image_size=28, channels=3, normalize=True):
        DatasetGenerator.__init__(self, path, batch_size, image_size, channels, normalize)
        self.dimensions = [28,28,1]
        self.load_dataset()

        print("Loaded MNIST from: ", path)

    def __num_classes__(self):
        return self.train[1].shape[0]

    def load_dataset(self):
        data = input_data.read_data_sets(self.path, one_hot=True)

        self.train = np.array([data.train.images, data.train.labels], dtype=np.float32)
        self.test = np.array([data.test.images, data.test.labels], dtype=np.float32)

        self.train[0] = np.asarray(np.reshape(self.train[0], (-1, 28, 28, 1))) / 255
        self.test[0] = np.asarray(np.reshape(self.test[0], (-1, 28, 28, 1))) / 255
        self.train = self.train.transpose()
        self.test = self.test.transpose()

        return

    def __getitem__(self, index):
        if index + 1 > self.__len__():
            self.batch_size = self.train.shape[0] - index * self.batch_size

        batch_x = self.train[index:index+1,0]
        batch_y = self.train[index:index+1,1]
        return batch_x, batch_y