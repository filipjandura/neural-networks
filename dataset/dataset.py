import numpy as np
import os
from tensorflow import keras


class DatasetGenerator(keras.utils.Sequence):
    def __init__(self, path, batch_size=8, image_size=28, channels=3, normalize=True):
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        self.normalize = normalize
        self.preprocessed = False
        self.train_x = []
        self.train_y = []
        self.train_cy = []
        self.test_x = []
        self.test_y = []
        self.test_cy = []

        self.train_idx = []
        self.test_idx = []

    def keep_indices(self, idxs):
        self.train_x = self.train_x[idxs]
        self.train_y = self.train_y[idxs]
        if len(self.train_cy) == len(idxs):
            self.train_cy = self.train_cy[idxs]
        self.train_idx = np.arange(len(self.train_x))

    def __load__(self, name, dir):
        pass
        return None

    def __getitem__(self, item):
        return None, None

    def on_epoch_end(self):
        np.random.shuffle(self.train_idx)
        np.random.shuffle(self.test_idx)

    def __num_classes__(self):
        return None

    def __len__(self):
        return len(self.train_idx) // self.batch_size

    def load_dataset(self):
        pass

    def __save_dataset(self):
        pass

    def _reset(self):
        self.train_x = []
        self.train_y = []

        self.test_x = []
        self.test_y = []
