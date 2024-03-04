from tensorflow.keras.datasets import cifar10
import numpy as np

def load_and_preprocess_data():
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train_gray = rgb2gray(x_train)
    x_test_gray = rgb2gray(x_test)
    return x_train, x_test, x_train_gray, x_test_gray

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).reshape(-1, 32, 32, 1)
