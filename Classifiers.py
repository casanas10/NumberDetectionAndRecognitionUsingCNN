import numpy as np
from tensorflow.contrib.keras.api import keras


def preprocess(X, y):

    X = np.asarray(X)
    X = X.astype('float32')
    X /= 255.0
    X = X.reshape(X.shape[0], 64, 64, 3)

    y = keras.utils.to_categorical(y, num_classes=11)
    y = [y[:, i] for i in range(5)]

    return X, y


def localizer_cnn():
    input_shape = (64, 64, 3)

    # BUILD the CNN
    cnn = keras.models.Sequential()

    cnn.add(keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    cnn.add(keras.layers.MaxPool2D())

    cnn.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    cnn.add(keras.layers.MaxPool2D())

    cnn.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    cnn.add(keras.layers.MaxPool2D())

    cnn.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    cnn.add(keras.layers.MaxPool2D())

    cnn.add(keras.layers.Flatten())
    cnn.add(keras.layers.Dense(1024, activation='relu'))
    cnn.add(keras.layers.Dropout(0.50))
    cnn.add(keras.layers.Dense(1024, activation='relu'))
    cnn.add(keras.layers.Dropout(0.50))
    cnn.add(keras.layers.Dense(1, activation='sigmoid'))

    # compile model
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # cnn.summary()

    return cnn

def custom_classifier():

    input_shape = (64, 64, 3)

    input = keras.layers.Input(shape=input_shape)

    x = keras.layers.BatchNormalization()(input)

    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv2D(192, (3, 3), activation='relu')(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.50)(x)

    y1 = keras.layers.Dense(11, activation='softmax')(x)
    y2 = keras.layers.Dense(11, activation='softmax')(x)
    y3 = keras.layers.Dense(11, activation='softmax')(x)
    y4 = keras.layers.Dense(11, activation='softmax')(x)
    y5 = keras.layers.Dense(11, activation='softmax')(x)

    model = keras.models.Model(inputs=input, outputs=[y1, y2, y3, y4, y5])

    learning_rate = 0.001
    decay_rate = learning_rate / 20
    momentum = 0.9
    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def vgg16():

    input_shape = (64, 64, 3)

    # BUILD the CNN
    input = keras.layers.Input(shape=input_shape)

    x = keras.layers.ZeroPadding2D((1, 1))(input)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPool2D((2, 2), strides=(2, 2))(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPool2D((2, 2), strides=(2, 2))(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPool2D((2, 2), strides=(2, 2))(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu')(x)
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu')(x)
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPool2D((2, 2), strides=(2, 2))(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu')(x)
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu')(x)
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPool2D((2, 2), strides=(2, 2))(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dropout(0.50)(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dropout(0.50)(x)

    y1 = keras.layers.Dense(11, activation='softmax')(x)
    y2 = keras.layers.Dense(11, activation='softmax')(x)
    y3 = keras.layers.Dense(11, activation='softmax')(x)
    y4 = keras.layers.Dense(11, activation='softmax')(x)
    y5 = keras.layers.Dense(11, activation='softmax')(x)

    model = keras.models.Model(inputs=input, outputs=[y1, y2, y3, y4, y5])

    learning_rate = 0.001
    decay_rate = learning_rate / 20
    momentum = 0.9
    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def pre_trained_vgg16():

    model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)

    input = keras.layers.Input(shape=(64, 64, 3))

    output = model(input)

    x = keras.layers.Flatten()(output)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dropout(0.50)(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dropout(0.50)(x)

    y1 = keras.layers.Dense(11, activation='softmax')(x)
    y2 = keras.layers.Dense(11, activation='softmax')(x)
    y3 = keras.layers.Dense(11, activation='softmax')(x)
    y4 = keras.layers.Dense(11, activation='softmax')(x)
    y5 = keras.layers.Dense(11, activation='softmax')(x)

    model = keras.models.Model(inputs=input, outputs=[y1, y2, y3, y4, y5])

    learning_rate = 0.001
    decay_rate = learning_rate / 20
    momentum = 0.9
    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model
