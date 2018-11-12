import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

INPUT_DIR = "input_images"

TRAINING_DIR = os.path.join(INPUT_DIR, "train_32x32.mat")
TEST_DIR = os.path.join(INPUT_DIR, "test_32x32.mat")


def load_images_and_split_data():
    data = sio.loadmat(TRAINING_DIR)

    X_train = data['X'][:, :, :, :40000].transpose()
    X_train = X_train.reshape(40000, 32, 32, 3)

    y_train = data['y'][:40000].flatten()
    y_train[y_train == 10] = 0

    X_test = data['X'][:, :, :, 10000:20000].transpose()
    X_test = X_test.reshape(10000, 32, 32, 3)

    y_test = data['y'][10000:20000].flatten()
    y_test[y_test == 10] = 0

    return X_train, y_train, X_test, y_test

