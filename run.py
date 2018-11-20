import os
import tensorflow as tf
import tensorflow.contrib.keras as keras

from tensorflow.contrib.keras.api import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2

import final_proj

if __name__ == "__main__":

    X_train, y_train, im_set = final_proj.load_dataset(number_of_images=2000)

    # X_train, mean = final_proj.preprocess_data(X_train)

    # X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

    # cv2.imshow('here', X_train[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    X_train = X_train.astype('float32')
    X_train /= 255.0

    input_shape = (244, 244, 3)

    print(X_train.shape)

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.35))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.35))

    model.add(keras.layers.Dense(4, activation='linear'))

    # input = keras.layers.Input(shape=(244, 244, 3))
    # x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input)
    # x = keras.layers.MaxPool2D()(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Dropout(0.25)(x)
    #
    # x = keras.layers.Flatten()(x)
    # x = keras.layers.Dense(512, activation='relu')(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Dropout(0.50)(x)
    #
    # x_bb = keras.layers.Dense(4, name='bb')(x)
    # # x_class = keras.layers.Dense(10, activation='softmax', name='class')(x)
    #
    # model = keras.models.Model([input], [x_bb])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

    history_cnn = model.fit(X_train, y_train, epochs=5, verbose=1, validation_data=(X_train, y_train))
    model.save_weights('model.h5')

    # model.load_weights('model.h5')

    # for i in range(len(X_train)):
    #     tt = X_train[i]
    #     #
    #     cv2.imshow('here', tt)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     te = np.asarray([tt])
    #     #
    #     image_s = im_set[i]
    #     cv2.imshow('here', image_s)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     #
    #     p = model.predict(te)
    #
    #     print(p[0])
    #     print(y_train[i])
    #
    #     x_min, x_max, y_min, y_max = final_proj.denormalize_bbox(y_train[i][0], y_train[i][1], y_train[i][2], y_train[i][3], image_s.shape[0], image_s.shape[1])
    #     #
    #     # x_min, x_max, y_min, y_max = final_proj.denormalize_bbox(p[0][0], p[0][1], p[0][2], p[0][3], image_s.shape[0], image_s.shape[1])
    #     # tt *= datagen.std
    #     # tt += datagen.mean
    #
    #     # x_min, x_max, y_min, y_max = y_train[i][0], y_train[i][1], y_train[i][2], y_train[i][3]
    #
    #     # x_min, x_max, y_min, y_max = int(p[0][0]), int(p[0][1]), int(p[0][2]), int(p[0][3])
    #
    #     im = final_proj.draw_bbox(tt, x_min, x_max, y_min, y_max)
    #
    #     cv2.imshow('here', im)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    pass
