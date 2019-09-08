import numpy as np
import os
import cv2
import json
from Classifiers import custom_classifier, localizer_cnn, vgg16, pre_trained_vgg16, preprocess

GRADED_IMAGES = "graded_images"
WEIGHTS_DIR = "weights"
LOCALIZER_WEIGHTS = os.path.join(WEIGHTS_DIR, "localizer_cnn.h5")
PRETRAINED_WEIGHTS = os.path.join(WEIGHTS_DIR, "vgg16_pre_trained.h5")

COUNT = 1

def load_images_from_dir(data_dir, ext=".png"):

    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    imagesFiles = sorted(imagesFiles, key=lambda x: int(os.path.splitext(x)[0]))
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f))) for f in imagesFiles]
    return imgs

def sliding_window(im, step, window):

    for j in range(0, im.shape[0] - window[1], step):
        for i in range(0, im.shape[1] - window[0], step):
            yield (i, j, im[j:j + window[1], i:i + window[0]])

def draw_bbox(im, x_min, y_min, x_max, y_max, number):

    cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
    cv2.putText(im, number, (x_min - 5, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255))
    return im

def find_sequence(image, k=None):

    #load the weights for both CNNs
    localizer = localizer_cnn()
    localizer.load_weights(LOCALIZER_WEIGHTS)

    pre_trained_cnn = pre_trained_vgg16()
    pre_trained_cnn.load_weights(PRETRAINED_WEIGHTS)

    max_bbox = (0, 0, 0, 0)
    max_p = 0
    digits = -1

    #resize image to (512, 512)
    image = cv2.resize(image, (512, 512))

    processed_image = image.copy()

    #preprocess
    image = image.astype('float32')
    image /= 255.0

    #square window sizes
    squares = [(28, 28), (48, 48), (64, 64), (128, 128), (256, 256)]

    regions = []
    bboxes = []

    for sq in squares:

        (winW, winH) = sq


        new_image = image.copy()

        for (x, y, window) in sliding_window(image, step=6, window=(winW, winH)):

            im2 = image[y:y + winH, x:x + winW]

            im2 = cv2.resize(im2, (64, 64))

            box = [x, y, x + winW, y + winH]

            regions.append(im2)
            bboxes.append(box)


    prediction = localizer.predict([regions])

    # print(prediction)

    # print(prediction, np.round(prediction))

    for i in range(len(prediction)):

        if prediction[i][0] > 0.95:

            # new_image = cv2.rectangle(new_image, (x, y), (x + winW, y + winH), (0, 255, 0), 1)
            #
            p2 = pre_trained_cnn.predict(np.asarray([regions[i]]))

            bbox = bboxes[i]

            sum = 0
            sequence = ''
            for p in p2:
                max_value = np.max(p)
                max_index = p[0].argmax()

                if max_value > 0.95 and max_index != 10:
                    sum += max_value
                    sequence += str(max_index)

            if max_p < sum:
                max_p = sum
                max_bbox = bbox
                digits = sequence
                # print(max_p)
                # print(max_bbox)
                # print('---------')

                # clone = image.copy()
                # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 1)
                # cv2.imshow("Window", clone)
                # cv2.waitKey(1)
                # time.sleep(0.0001)

    processed_image = draw_bbox(processed_image, max_bbox[0], max_bbox[1], max_bbox[2], max_bbox[3], digits)

    # processed_image = np.uint8(processed_image * 255.0)

    if k == None:
        return processed_image
    else:
        cv2.imwrite(os.path.join(GRADED_IMAGES, (str(k) + '.png')), processed_image)

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

def load_dataset(images, file=None, number_of_images=None):

    temp_im = []

    if file == None:
        for i in range(len(images)):

#             im = cv2.resize(images[i], (64, 64))

            temp_im.append(images[i])
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

def split_dataset(training_images, testing_images, train_file=None, test_file=None):

    X_train, y_train = cropbbox(training_images, train_file)

    X_test, y_test = cropbbox(testing_images, test_file)

    return X_train, X_test, y_train, y_test
