import os
from Classifiers import custom_classifier, localizer_cnn, vgg16, pre_trained_vgg16, preprocess
import utils
import tensorflow.contrib.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

INPUT_DIR = "input_images"
OUTPUT_DIR = "output"
GRADED_IMAGES = "graded_images"
WEIGHTS_DIR = "weights"
VID_DIR = "input_videos"

TRAINING_DIR = os.path.join(INPUT_DIR, "train")
TESTING_DIR = os.path.join(INPUT_DIR, "test")
NEG_DIR = os.path.join(INPUT_DIR, "neg")
TA_TESTING_FOLDER = os.path.join(INPUT_DIR, "TA_Testing_Folder")
GENERATED_IMAGES = os.path.join(INPUT_DIR, "generated_images")

LOCALIZER_WEIGHTS = os.path.join(WEIGHTS_DIR, "localizer_cnn.h5")
CUSTOM_CLASSIFIER_WEIGHTS = os.path.join(WEIGHTS_DIR, "custom_model.h5")
VGG16_WEIGHTS = os.path.join(WEIGHTS_DIR, "vgg16.h5")
PRETRAINED_WEIGHTS = os.path.join(WEIGHTS_DIR, "vgg16_pre_trained.h5")

TRAINING_FILE = os.path.join(TRAINING_DIR, "digitStruct.json")
TESTING_FILE = os.path.join(TRAINING_DIR, "digitStruct.json")


#Binary classifier to determine if the image has digits or not
def run_localizer(weights=None):

    #load pos and negative images
    data = utils.load_images_from_dir(TRAINING_DIR)
    background = utils.load_images_from_dir(NEG_DIR)

    pos = utils.load_dataset(data, TRAINING_FILE)[:27000]
    neg = utils.load_dataset(background)[:45000]

    X_train = pos + neg
    X_train = np.asarray(X_train, dtype=np.float32)
    X_train /= 255.0
    X_train = X_train.reshape(X_train.shape[0], 64, 64, 3)

    labels = np.array(len(pos) * [1] + len(neg) * [0])

    #split data train 80% , test 20%
    X_train, X_test, y_train, y_test = train_test_split(X_train, labels, test_size=0.2, train_size=0.8)

    #build model
    cnn = localizer_cnn()

    #if no weights train it else load the weights
    if weights==None:

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
                                    epochs=15,
                                    validation_data=(X_train, y_train))

        cnn.save_weights(os.path.join(WEIGHTS_DIR, 'localizer_cnn.h5'))
    else:

        cnn.load_weights(weights)

    #evaluate the model
    score = cnn.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

#custom number classifier to determine which digits are in the image
def run_custom_classifier(weights=None):

    training_images = utils.load_images_from_dir(TRAINING_DIR)
    testing_images = utils.load_images_from_dir(TESTING_DIR)

    X_train, X_test, y_train, y_test = utils.split_dataset(training_images, testing_images, TRAINING_FILE, TESTING_FILE)

    #preprocess data
    X_train, y_train = preprocess(X_train, y_train)
    X_test, y_test = preprocess(X_test, y_test)

    #compile model
    model = custom_classifier()

    #train if no weights are passed in
    if weights == None:

        history = model.fit(X_train, y_train, epochs=50, verbose=1, validation_data=(X_test, y_test))

        model.save_weights(os.path.join(WEIGHTS_DIR, 'custom_model.h5'))

    else:
        model.load_weights(weights)


    scores = model.evaluate(X_train, y_train, verbose=1)
    print('Digit 1 loss:', scores[1])
    print('Digit 2 loss:', scores[2])
    print('Digit 3 loss:', scores[3])
    print('Digit 4 loss:', scores[4])
    print('Digit 5 loss:', scores[5])
    average_loss = sum([scores[i] for i in range(1, 6)]) / 5
    print('Average loss:', average_loss)

    print('Digit 1 accuracy:', scores[6])
    print('Digit 2 accuracy:', scores[7])
    print('Digit 3 accuracy:', scores[8])
    print('Digit 4 accuracy:', scores[9])
    print('Digit 5 accuracy:', scores[10])

    average_accuracy = sum([scores[i] for i in range(6, 11)]) / 5
    print('Average accueracy:', average_accuracy)

def run_vgg16(weights=None):

    training_images = utils.load_images_from_dir(TRAINING_DIR)
    testing_images = utils.load_images_from_dir(TESTING_DIR)

    X_train, X_test, y_train, y_test = utils.split_dataset(training_images, testing_images, TRAINING_FILE, TESTING_FILE)

    # preprocess data
    X_train, y_train = preprocess(X_train, y_train)
    X_test, y_test = preprocess(X_test, y_test)

    # compile model
    model = vgg16()

    # train if no weights are passed in
    if weights == None:

        history = model.fit(X_train, y_train, epochs=50, verbose=1, validation_data=(X_test, y_test))

        model.save_weights(os.path.join(WEIGHTS_DIR, 'vgg16.h5'))

    else:
        model.load_weights(weights)

    scores = model.evaluate(X_train, y_train, verbose=1)
    print('Digit 1 loss:', scores[1])
    print('Digit 2 loss:', scores[2])
    print('Digit 3 loss:', scores[3])
    print('Digit 4 loss:', scores[4])
    print('Digit 5 loss:', scores[5])
    average_loss = sum([scores[i] for i in range(1, 6)]) / 5
    print('Average loss:', average_loss)

    print('Digit 1 accuracy:', scores[6])
    print('Digit 2 accuracy:', scores[7])
    print('Digit 3 accuracy:', scores[8])
    print('Digit 4 accuracy:', scores[9])
    print('Digit 5 accuracy:', scores[10])

    average_accuracy = sum([scores[i] for i in range(6, 11)]) / 5
    print('Average accueracy:', average_accuracy)


def run_pretrained_vgg16(weights=None):
    training_images = utils.load_images_from_dir(TRAINING_DIR)
    testing_images = utils.load_images_from_dir(TESTING_DIR)

    X_train, X_test, y_train, y_test = utils.split_dataset(training_images, testing_images, TRAINING_FILE, TESTING_FILE)

    # preprocess data
    X_train, y_train = preprocess(X_train, y_train)
    X_test, y_test = preprocess(X_test, y_test)

    # compile model
    model = pre_trained_vgg16()

    # train if no weights are passed in
    if weights == None:

        history = model.fit(X_train, y_train, epochs=50, verbose=1, validation_data=(X_test, y_test))

        model.save_weights(os.path.join(WEIGHTS_DIR, 'vgg16_pre_trained.h5'))

    else:
        model.load_weights(weights)

    scores = model.evaluate(X_train, y_train, verbose=1)
    print('Digit 1 loss:', scores[1])
    print('Digit 2 loss:', scores[2])
    print('Digit 3 loss:', scores[3])
    print('Digit 4 loss:', scores[4])
    print('Digit 5 loss:', scores[5])
    average_loss = sum([scores[i] for i in range(1, 6)]) / 5
    print('Average loss:', average_loss)

    print('Digit 1 accuracy:', scores[6])
    print('Digit 2 accuracy:', scores[7])
    print('Digit 3 accuracy:', scores[8])
    print('Digit 4 accuracy:', scores[9])
    print('Digit 5 accuracy:', scores[10])

    average_accuracy = sum([scores[i] for i in range(6, 11)]) / 5
    print('Average accueracy:', average_accuracy)


def process_images(folder):
    images = utils.load_images_from_dir(folder)  # make sure images are .png
    k = 1
    for im in images:
        print("Processing image {}".format(k))
        utils.find_sequence(im, k)
        k += 1

def process_videos(filename):

    fps = 40

    vidcap = cv2.VideoCapture(filename)
    ret, frame = vidcap.read()

    out_path = "output/test_output.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_out = cv2.VideoWriter(out_path, fourcc, fps, (512, 512))

    frame_num = 1
    while ret:

        print("Processing frame {}".format(frame_num))

        ret, frame = vidcap.read()

        if ret:
            process_image = utils.find_sequence(frame)

            video_out.write(process_image)

            frame_num += 1

    vidcap.release()

def run_data_augmentation():


    data = utils.load_images_from_dir(TRAINING_DIR)
    data = np.array(utils.load_dataset(data, TESTING_FILE))

    datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                        width_shift_range=0.2,
                                                        height_shift_range=0.2,
                                                        fill_mode='nearest',
                                                        )
    datagen.fit(data)
    for X_batch in datagen.flow(data, batch_size=9, save_to_dir=GENERATED_IMAGES, save_prefix='aug', save_format='png'):
        # for i in range(0, 9):
        #     plt.subplot(330 + 1 + i)
        #     plt.imshow((X_batch[i]).astype(np.uint8))
        # plt.show()
        break

if __name__ == "__main__":

    #DATA AUGMENTATION
    # run_data_augmentation()

    #RUN LOCALIZER
    # run_localizer(LOCALIZER_WEIGHTS)

    #RUN MY CUSTOM ARCHITECTURE
    # run_custom_classifier(CUSTOM_CLASSIFIER_WEIGHTS)

    #RUN VGG16 
    # run_vgg16(VGG16_WEIGHTS)
    
    #RUN VGG16 WITH PRE_TRAINED WEIGHTS
    # run_pretrained_vgg16(PRETRAINED_WEIGHTS)

    #IMAGE PROCESSING
    process_images(TA_TESTING_FOLDER)

    #VIDEO PROCESSING
    # video_file = "test.mp4"
    # video = os.path.join(VID_DIR, video_file)
    # process_videos(video)

