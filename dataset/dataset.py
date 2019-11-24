import numpy as np
import os
from tensorflow import keras
import cv2


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
        self.test_x = []
        self.test_y = []

        self.train_idx = []
        self.test_idx = []

    def __load__(self, name, dir):
        ## Path
        abs_img_path = os.path.join(self.path, dir, name)
        ## Reading Image
        image = cv2.imread(abs_img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if not image.shape == (self.image_size, self.image_size, self.channels):
            image = cv2.resize(image, (self.image_size, self.image_size))
        ## Normalizaing
        image = image / 0xffff
        return image

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