import numpy as np
from skimage.color import rgb2hsv


def zca_whitening(data:np.ndarray, epsilon:float=0.1):
    x = data.reshape(data.shape[0], data.shape[1]*data.shape[2]*data.shape[3])
    # global contrast normalization  / np.std(x)
    x = (x - np.mean(x, axis=0))
    # ZCA whitening
    cov = np.cov(x, rowvar=True)
    u,s,v = np.linalg.svd(cov)
    x_zca = u.dot(np.diag(1.0/np.sqrt(s+epsilon))).dot(u.transpose()).dot(x)
    x_zca_rescaled = (x_zca - x_zca.min()) / (x_zca.max() - x_zca.min())
    return x_zca_rescaled.reshape(data.shape)

def map_rgb2hsv(data:np.ndarray):
    return np.array(list(map(rgb2hsv, data)))