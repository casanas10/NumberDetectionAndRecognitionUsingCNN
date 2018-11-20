import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

INPUT_DIR = "input_images"
TRAINING_DIR = os.path.join(INPUT_DIR, "train")
TRAINING_FILE = os.path.join(TRAINING_DIR, "digitStruct.json")

image_mean = 0
image_std = 0

def load_images_from_dir(data_dir, ext=".png"):

    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    imagesFiles = sorted(imagesFiles, key=lambda x: int(os.path.splitext(x)[0]))
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f))) for f in imagesFiles]
    # imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    return imgs

def load_dataset(number_of_images=None):

    im_set = []

    images = load_images_from_dir(TRAINING_DIR)[:number_of_images]
    temp_im = []

    temp_lb = []

    with open(TRAINING_FILE) as f:

        data = json.load(f)

        for i in range(len(images)):
            # plt.imshow(images[i])
            # plt.show()

            bbox = data[i]['boxes']

            if len(bbox) <= 2:

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

                # expanded_left_min = int(left_min - (left_min * .05))
                # expanded_top_min = int(top_min - (top_min * .05))
                # expanded_right_max = int((right_max + (right_max * .05)))
                # expanded_bottom_max = int((bottom_max + (bottom_max * .05)))
                #
                # crop_img = images[i][expanded_top_min:expanded_bottom_max, expanded_left_min:expanded_right_max]
                #
                # cv2.imshow("cropped", crop_img)
                # cv2.waitKey(0)

                # crop_img = np.reshape(crop_img, (48, 48))

                x_scale = 244 / images[i].shape[1]
                y_scale = 244 / images[i].shape[0]

                left_min = int(np.round(left_min * x_scale))
                top_min = int(np.round(top_min * y_scale))
                right_max = int(np.round(right_max * x_scale))
                bottom_max = int(np.round(bottom_max * y_scale))

                x, y, w, h = normalize_box(left_min, top_min, right_max, bottom_max, images[i].shape[0],
                                           images[i].shape[1])

                # x = left_min
                # y = top_min
                # w = right_max - left_min
                # h = bottom_max - top_min

                crop_img = cv2.resize(images[i], (244, 244))

                # im = draw_bbox(crop_img, x, x + w, y, y + h)
                # cv2.imshow("cropped", im)
                # cv2.waitKey(0)



                temp_im.append(crop_img)
                temp_lb.append([x, y, w, h])

                im_set.append(images[i])


    # images = [cv2.resize(x, (48, 48)) for x in temp_im]

    X_train = np.asarray(temp_im)
    y_train = np.asarray(temp_lb)

    return X_train, y_train, im_set

def normalize_box(x_min, x_max, y_min, y_max, image_width, image_height):

    x = ((x_min + x_max) / 2.0) * (1.0 / image_width)
    y = ((y_min + y_max) / 2.0) * (1.0 / image_height)
    w = (x_max - x_min) * (1.0 / image_width)
    h = (y_max - y_min) * (1.0 / image_height)

    return x, y, w, h

def denormalize_bbox(x, y, w, h, image_width, image_height):

    x_min = int((x * image_width) - ((w * image_width) / 2.0))
    x_max = int((x * image_width) + ((w * image_width) / 2.0))
    y_min = int((y * image_height) - ((h * image_height) / 2.0))
    y_max = int((y * image_height) + ((h * image_height) / 2.0))

    return x_min, x_max, y_min, y_max

def draw_bbox(im, x_min, x_max, y_min, y_max):

    x_min = int(x_min)
    x_max = int(x_max)
    y_min = int(y_min)
    y_max = int(y_max)

    cv2.rectangle(im, (x_min, x_max), (y_min, y_max), (0, 255, 0), 1)
    # cv2.putText(im, 'Number', (x_min + x_max + 10, y_min + y_max), 0, 0.3, (0, 255, 0))
    return im
