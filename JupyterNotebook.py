
# coding: utf-8

# In[1]:


import os
import tensorflow.contrib.keras as keras

from tensorflow.contrib.keras.api import keras
import numpy as np
import cv2
import json


# In[2]:


INPUT_DIR = "input_images"
OUTPUT_DIR = "graded_images"

TRAINING_DIR = os.path.join(INPUT_DIR, "train")
TESTING_DIR = os.path.join(INPUT_DIR, "test")
EXTRA_DIR = os.path.join(INPUT_DIR, "extra")

EXTRA_FILE = os.path.join(EXTRA_DIR, "digitStruct.json")
G_FOLDER = os.path.join(INPUT_DIR, "g")
TRAINING_FILE = os.path.join(TRAINING_DIR, "digitStruct.json")
TESTING_FILE = os.path.join(TESTING_DIR, "digitStruct.json")
NEG_DIR = os.path.join(INPUT_DIR, "neg")


# In[3]:


def load_images_from_dir(data_dir, ext=".png"):

    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    imagesFiles = sorted(imagesFiles, key=lambda x: int(os.path.splitext(x)[0]))
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f))) for f in imagesFiles]
    return imgs


# In[4]:


def cropbbox(images, file=None):

    X = []
    y = []

    with open(file) as f:

        data = json.load(f)

        for i in range(len(images)):

            try:

                bbox = data[i]['boxes']

                if len(bbox) <= 5:
                    left_min = 10000
                    top_min = 10000
                    right_max = -1
                    bottom_max = -1

                    label = [10, 10, 10, 10, 10]

                    k = 0
                    for j in bbox:

                        l = int(j['label'])

                        if l == 10:
                            label[k] = 0
                        else:
                            label[k] = l

                        if left_min > j['left']:
                            left_min = j['left']
                        if top_min > j['top']:
                            top_min = j['top']
                        if right_max < j['left'] + j['width']:
                            right_max = j['left'] + j['width']
                        if bottom_max < j['top'] + j['height']:
                            bottom_max = j['top'] + j['height']

                        k += 1

                    expanded_left_min = int(left_min - (left_min * .15))
                    expanded_top_min = int(top_min - (top_min * .15))
                    expanded_right_max = int((right_max + (right_max * .15)))
                    expanded_bottom_max = int((bottom_max + (bottom_max * .15)))

                    crop_img = images[i][expanded_top_min:expanded_bottom_max, expanded_left_min:expanded_right_max]

                    im = cv2.resize(crop_img, (64, 64))

                    X.append(im)
                    y.append(label)


            except:
                label = [10, 10, 10, 10, 10]

                im = cv2.resize(images[i], (64, 64))

                X.append(im)
                y.append(label)

    return X, y


# In[5]:


def load_dataset(images, file=None, number_of_images=None):
    im_set = []
    temp_im = []
    temp_lb = []

    if file == None:
        for i in images:

#             im = cv2.resize(i, (64, 64))

            temp_im.append(i)
    else:

        with open(file) as f:

            data = json.load(f)

            for i in range(len(images)):

                    bbox = data[i]['boxes']

                    if len(bbox) <= 5:

                        left_min = 10000
                        top_min = 10000
                        right_max = -1
                        bottom_max = -1

                        bbox = data[i]['boxes']
                        for j in bbox:

                            if left_min > j['left']:
                                left_min = j['left']
                            if top_min > j['top']:
                                top_min = j['top']
                            if right_max < j['left'] + j['width']:
                                right_max = j['left'] + j['width']
                            if bottom_max < j['top'] + j['height']:
                                bottom_max = j['top'] + j['height']

                        expanded_left_min = int(left_min - (left_min * .05))
                        expanded_top_min = int(top_min - (top_min * .05))
                        expanded_right_max = int((right_max + (right_max * .05)))
                        expanded_bottom_max = int((bottom_max + (bottom_max * .05)))

                        crop_img = images[i][expanded_top_min:expanded_bottom_max, expanded_left_min:expanded_right_max]

                        im = cv2.resize(crop_img, (64, 64))

                        temp_im.append(im)


    return temp_im


# In[6]:


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

    cnn.summary()

    return cnn


# In[7]:


# train the localalizer classifier

data = load_images_from_dir(TRAINING_DIR)
background = load_images_from_dir(NEG_DIR)

pos = load_dataset(data, TRAINING_FILE)
neg = load_dataset(background)

print(len(pos))
print(len(neg))

X_train = pos + neg
X_train = np.asarray(X_train, dtype=np.float32)
X_train /= 255.0
X_train = X_train.reshape(X_train.shape[0], 64, 64, 3)

y_train = np.array(len(pos) * [1] + len(neg) * [0])

cnn = localizer_cnn()

datagen = keras.preprocessing.image.ImageDataGenerator(
                                                               rotation_range=40,
                                                               width_shift_range=0.2,
                                                               height_shift_range=0.2,
                                                               zoom_range=0.2,
                                                               fill_mode='nearest',
                                                               )
datagen.fit(X_train)
history = cnn.fit_generator(datagen.flow(X_train, y_train, batch_size=64),
                                steps_per_epoch=X_train.shape[0] // 64,
                                epochs=25,
                                validation_data=(X_train, y_train))

cnn.save_weights('localizer_cnn.h5')


# In[8]:


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


# In[9]:



# TRAINING DATA
images = load_images_from_dir(TRAINING_DIR)
X_train, y_train = cropbbox(images, TRAINING_FILE)

X_train = np.asarray(X_train)
X_train = X_train.astype('float32')
X_train /= 255.0
X_train = X_train.reshape(X_train.shape[0], 64, 64, 3)

y_train = keras.utils.to_categorical(y_train, num_classes=11)
y_train = [y_train[:, i] for i in range(5)]

#TESTING DATA
images = load_images_from_dir(TESTING_DIR)
X_test, y_test = cropbbox(images, TESTING_FILE)

X_test = np.asarray(X_test)
X_test = X_test.astype('float32')
X_test /= 255.0
X_test = X_test.reshape(X_test.shape[0], 64, 64, 3)

y_test = keras.utils.to_categorical(y_test, num_classes=11)
y_test = [y_test[:, i] for i in range(5)]

model = custom_classifier()

learning_rate = 0.001
decay_rate = learning_rate / 20
momentum = 0.9
sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, verbose=1, batch_size=250,
          validation_data=(X_test, y_test))

model.save_weights('custom_model.h5')


# In[12]:


print(history.history.keys())   

# summarize history for accuracy  

# plt.plot(history.history['val_dense_1_acc'], label = '1')
# plt.plot(history.history['val_dense_2_acc'], label = '2')
# plt.plot(history.history['val_dense_3_acc'], label = '3')
# plt.plot(history.history['val_dense_4_acc'], label = '4')
# plt.plot(history.history['val_dense_5_acc'], label = '5')
# plt.title('Model Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['1', '2', '3','4','5'], loc='upper left')
# plt.show()
#
#
# plt.plot(history.history['val_dense_1_loss'], label = '1')
# plt.plot(history.history['val_dense_2_loss'], label = '2')
# plt.plot(history.history['val_dense_3_loss'], label = '3')
# plt.plot(history.history['val_dense_4_loss'], label = '4')
# plt.plot(history.history['val_dense_5_loss'], label = '5')
# plt.title('Model Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['1', '2', '3','4','5'], loc='upper left')
# plt.show()

model.load_weights('custom_model.h5')

score = model.evaluate(X_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[13]:


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


# In[14]:


# TRAINING DATA
images = load_images_from_dir(TRAINING_DIR)
X_train, y_train = cropbbox(images, TRAINING_FILE)

X_train = np.asarray(X_train)
X_train = X_train.astype('float32')
X_train /= 255.0
X_train = X_train.reshape(X_train.shape[0], 64, 64, 3)

y_train = keras.utils.to_categorical(y_train, num_classes=11)
y_train = [y_train[:, i] for i in range(5)]

#TESTING DATA
images = load_images_from_dir(TESTING_DIR)
X_test, y_test = cropbbox(images, TESTING_FILE)

X_test = np.asarray(X_test)
X_test = X_test.astype('float32')
X_test /= 255.0
X_test = X_test.reshape(X_test.shape[0], 64, 64, 3)

y_test = keras.utils.to_categorical(y_test, num_classes=11)
y_test = [y_test[:, i] for i in range(5)]

model = vgg16()

learning_rate = 0.001
decay_rate = learning_rate / 20
momentum = 0.9
sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, verbose=1, batch_size=250,
          validation_data=(X_test, y_test))

model.save_weights('vgg16.h5')


# In[ ]:


# plt.plot(history.history['val_dense_2_acc'], label = '1')
# plt.plot(history.history['val_dense_3_acc'], label = '2')
# plt.plot(history.history['val_dense_4_acc'], label = '3')
# plt.plot(history.history['val_dense_5_acc'], label = '4')
# plt.plot(history.history['val_dense_6_acc'], label = '5')
# plt.title('Model Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['1', '2', '3','4','5'], loc='upper left')
# plt.show()


# plt.plot(history.history['val_dense_2_loss'], label = '1')
# plt.plot(history.history['val_dense_3_loss'], label = '2')
# plt.plot(history.history['val_dense_4_loss'], label = '3')
# plt.plot(history.history['val_dense_5_loss'], label = '4')
# plt.plot(history.history['val_dense_6_loss'], label = '5')
# plt.title('Model Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['1', '2', '3','4','5'], loc='upper left')
# plt.show()


# In[17]:


# # vgg16 training

images = load_images_from_dir(TRAINING_DIR)

# TRAINING DATA
X_train, y_train = cropbbox(images, TRAINING_FILE)

X_train = np.asarray(X_train)
X_train = X_train.astype('float32')
X_train /= 255.0
X_train = X_train.reshape(X_train.shape[0], 64, 64, 3)

y_train = keras.utils.to_categorical(y_train, num_classes=11)
y_train = [y_train[:, i] for i in range(5)]

#TESTING DATA
images = load_images_from_dir(TESTING_DIR)
X_test, y_test = cropbbox(images, TESTING_FILE)

X_test = np.asarray(X_test)
X_test = X_test.astype('float32')
X_test /= 255.0
X_test = X_test.reshape(X_test.shape[0], 64, 64, 3)

y_test = keras.utils.to_categorical(y_test, num_classes=11)
y_test = [y_test[:, i] for i in range(5)]

model = keras.applications.vgg16.VGG16(weights=None, include_top=False)

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
model.summary()

history = model.fit(X_train, y_train, epochs=50, verbose=1, batch_size=250,
          validation_data=(X_test, y_test))

model.save_weights('vgg16.h5')

print(history.history.keys())   

# summarize history for accuracy  

# plt.plot(history.history['val_dense_2_acc'], label = '1')
# plt.plot(history.history['val_dense_3_acc'], label = '2')
# plt.plot(history.history['val_dense_4_acc'], label = '3')
# plt.plot(history.history['val_dense_5_acc'], label = '4')
# plt.plot(history.history['val_dense_6_acc'], label = '5')
# plt.title('Model Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['1', '2', '3','4','5'], loc='upper left')
# plt.show()
#
#
# plt.plot(history.history['val_dense_2_loss'], label = '1')
# plt.plot(history.history['val_dense_3_loss'], label = '2')
# plt.plot(history.history['val_dense_4_loss'], label = '3')
# plt.plot(history.history['val_dense_5_loss'], label = '4')
# plt.plot(history.history['val_dense_6_loss'], label = '5')
# plt.title('Model Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['1', '2', '3','4','5'], loc='upper left')
# plt.show()

model.load_weights('vgg16.h5')

model.evaluate(X_test, y_test)

