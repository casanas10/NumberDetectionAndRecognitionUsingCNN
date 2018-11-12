import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

INPUT_DIR = "input_images"

TRAINING_DIR = os.path.join(INPUT_DIR, "train_32x32.mat")
TEST_DIR = os.path.join(INPUT_DIR, "test_32x32.mat")


def load_images_and_split_data():
    data = sio.loadmat(TRAINING_DIR)

    X_train = data['X'][:, :, :, :50]
    y_train = data['y'][:50]

    X_test = data['X'][:, :, :, 50:100]
    y_test = data['y'][50:100]

    return X_train, y_train, X_test, y_test


