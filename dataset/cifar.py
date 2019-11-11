from dataset.dataset import DatasetGenerator
import numpy as np
import os
import pickle

class DatasetCIFAR(DatasetGenerator):
    def __init__(self, path, batch_size, image_size=28, channels=3, normalize=True):
        print('Loading Dataset CIFAR-10 from ' + path)
        DatasetGenerator.__init__(self, path, batch_size, image_size, channels, normalize)

        self.classes = 10
        self.preprocessed = False
        self.load_dataset()
        if self.preprocessed == False:
            self.preprocess()
            self.__save_dataset()

        self.dimensions = [32, 32, 3]
        self.train_data_length = self.train.shape[0]

        print("Loaded CIFAR-10 from: ", path)

    def load_dataset(self):
        files = os.listdir(self.path)

        if 'preprocessed_train' in files:
            file = os.path.join(self.path, 'preprocessed_train')
            data, labels = self.__unpickle(file)
            self.train = np.array([data, labels]).transpose()

            file = os.path.join(self.path, 'preprocessed_test')
            data, labels = self.__unpickle(file)
            self.test = np.array([data, labels]).transpose()

            print('Loaded {:d} train images and {:d} test images'.format(self.train.shape[0], self.test.shape[0]))
        else:
            if len(files) > 0:
                for f in files:
                    file = os.path.join(self.path, f)
                    data, labels = self.__unpickle(file)
                    if not 'test' in file.lower():
                        if len(self.train) == 0:
                            self.train = np.array([data, labels])
                        else:
                            self.train = np.concatenate((self.train_x, np.array([data, labels])), axis=0)
                    else:
                        self.test = np.array([data, labels], dtype=np.float32)

            zeros = np.zeros((len(self.train_y), self.classes), dtype=np.float32)
            zeros[np.arange(len(self.train_y)), self.train_y] = 1
            self.train_y = zeros

            zeros = np.zeros((len(self.test_y), self.classes), dtype=np.float32)
            zeros[np.arange(len(self.test_y)), self.test_y] = 1
            self.test_y = zeros

            self.train_x = self.train_x.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1]) / 255
            self.test_x = self.test_x.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1]) / 255

    def preprocess(self, batch=2000):
        ## Preprocess data
        through_data = int(self.train_x.shape[0] / batch)
        for i in range(through_data):
            print("Normalizing train {:d}-{:d}/{:d}".format(i * batch, (i + 1) * batch, self.train_x.shape[0]))
            data = self.train_x[i * batch: (i + 1) * batch]
            self.train_x[i * batch: (i + 1) * batch] = self.__zca_whitening(data)
        through_data = int(self.test_x.shape[0] / batch)
        for i in range(through_data):
            print("Normalizing test {:d}-{:d}/{:d}".format(i * batch, (i + 1) * batch, self.test_x.shape[0]))
            data = self.test_x[i * batch: (i + 1) * batch]
            self.test_x[i * batch: (i + 1) * batch] = self.__zca_whitening(data)
        print("Data normalized by ZCA whitening")

        self.train_x = np.concatenate((self.train_x, np.flip(self.train_x, 1)))
        self.train_y = np.concatenate((self.train_y, self.train_y))
        self.preprocessed = True

    def __save_dataset(self):
        data_train = {b'data':self.train_x, b'labels':self.train_y}
        data_test = {b'data':self.test_x, b'labels':self.test_y}

        with open(os.path.join(self.path,'preprocessed_train'),mode='wb') as file:
            pickle.dump(data_train, file, pickle.DEFAULT_PROTOCOL)
        with open(os.path.join(self.path,'preprocessed_test'), mode='wb') as file:
            pickle.dump(data_test, file, pickle.DEFAULT_PROTOCOL)

    def __unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict[b'data'], dict[b'labels']

    def __zca_whitening(self, data:np.ndarray, epsilon:float=0.1):
        x = data.reshape(data.shape[0], data.shape[1]*data.shape[2]*data.shape[3])
        # global contrast normalization  / np.std(x)
        x = (x - np.mean(x, axis=0))
        # ZCA whitening
        cov = np.cov(x, rowvar=True)
        u,s,v = np.linalg.svd(cov)
        x_zca = u.dot(np.diag(1.0/np.sqrt(s+epsilon))).dot(u.transpose()).dot(x)
        x_zca_rescaled = (x_zca - x_zca.min()) / (x_zca.max() - x_zca.min())
        return x_zca_rescaled.reshape(data.shape)

