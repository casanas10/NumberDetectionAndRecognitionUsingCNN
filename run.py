import cv2
import os
import keras
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt

import final_proj


def run():

    # load images and split data set into train, validation, and test
    X_train, y_train, X_test, y_test = final_proj.load_images_and_split_data()

    print(X_train.shape)
    print(y_train.shape)

    #preprocess data
    num_classes = 10
    epochs = 10

    input_shape = (32, 32, 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    #rescale data so between 0 and 1
    X_train /= 255.0
    X_test /= 255.0

    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    print(X_train.shape)
    print(y_train.shape)

    # BUILD the CNN
    cnn = Sequential()

    cnn.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    cnn.add(Conv2D(32, (3, 3), activation='relu'))
    cnn.add(MaxPool2D())
    cnn.add(Dropout(0.25))

    cnn.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    cnn.add(Conv2D(32, (3, 3), activation='relu'))
    cnn.add(MaxPool2D())
    cnn.add(Dropout(0.25))

    cnn.add(Flatten()) #we flatten the network because we have a Dense(Fully Connected layer next)
    cnn.add(Dense(1024, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(10, activation='softmax')) #output layer

    #compile model
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(cnn.summary())

    history_cnn = cnn.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_train, y_train))

    # Convert labels to categorical one-hot encoding
    # one_hot_labels = to_categorical(y_train)
    #
    # history_cnn = cnn.fit(X_train, y_train, epochs)

    print(history_cnn.history['acc'])

    #
    # plt.plot(history_cnn.history['acc'])
    # plt.show()
    #
    # #you can also load weights from other trained models
    # # cnn.load_weights('weights/cnn-model5.h5')
    #
    #
    # #evalute model
    # score = cnn.evaluate(X_test, y_test)
    #
    # print(score)


    # define hyper parameters

    # TRAINING

    # train model

    # get predictions

    # calculate accuracy of training

    # EVALUATE

    # TESTING

    pass


if __name__ == "__main__":
    run()

    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #
    # # print(X_train.shape)
    # # print(y_train.shape)
    #
    # # plt.imshow(X_train[0])
    # # plt.show()
    # #
    # # print(y_train[0])
    #
    # #preprocessing
    # num_classes = 10
    # epochs = 3
    #
    # X_train = X_train.reshape(60000, 28, 28, 1)
    # X_test = X_test.reshape(10000, 28, 28, 1)
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # #rescale data so between 0 and 1
    # X_train /= 255.0
    # X_test /= 255.0
    # y_train = to_categorical(y_train, num_classes)
    # y_test = to_categorical(y_test, num_classes)
    #
    # print(X_train.shape)
    # print(y_train.shape)
    #
    # #create and compile the model
    # cnn = Sequential()
    # cnn.add(Conv2D(32, kernel_size=(5,5), input_shape=(28, 28, 1), padding='same', activation='relu'))
    # cnn.add(MaxPool2D())
    # cnn.add(Conv2D(64, kernel_size=(5,5), padding='same', activation='relu'))
    # cnn.add(MaxPool2D())
    # cnn.add(Flatten()) #we flatten the network because we have a Dense(Fully Connected layer next)
    # cnn.add(Dense(1024, activation='relu'))
    # cnn.add(Dense(10, activation='softmax')) #output layer
    #
    # #compile model
    # cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # print(cnn.summary())
    #
    #
    # #train model
    # history_cnn = cnn.fit(X_train, y_train, epochs, verbose=1, validation_data=(X_train, y_train))
    #
    # plt.plot(history_cnn.history['acc'])
    # plt.show()
    #
    # #you can also load weights from other trained models
    # # cnn.load_weights('weights/cnn-model5.h5')
    #
    #
    # #evalute model
    # score = cnn.evaluate(X_test, y_test)
    #
    # print(score)

    # import numpy as np
    # import keras
    # from keras.models import Sequential
    # from keras.layers import Dense, Dropout, Flatten
    # from keras.layers import Conv2D, MaxPooling2D
    # from keras.optimizers import SGD
    #
    # # Generate dummy data
    # x_train = np.random.random((100, 100, 100, 3))
    #
    # y = np.random.randint(10, size=(100, 1))
    #
    # y_train = keras.utils.to_categorical(y, num_classes=10)
    # x_test = np.random.random((20, 100, 100, 3))
    # y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
    #
    # model = Sequential()
    # # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # # this applies 32 convolution filters of size 3x3 each.
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))
    #
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd)
    #
    # model.fit(x_train, y_train, batch_size=32, epochs=10)
    # score = model.evaluate(x_test, y_test, batch_size=32)