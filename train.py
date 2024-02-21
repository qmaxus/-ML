import os
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras_segmentation.models.model_utils import get_segmentation_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 0-GPU -1-CPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
session = tf.compat.v1.keras.backend.get_session()


def get_image_arr(path, width, height, imgNorm="sub_mean", odering='channels_first'):
    if type(path) is np.ndarray:
        img = path
    else:
        img = cv2.imread(path, 1)

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img / 255.0

    if odering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def predict_my(model=None, inp=None):
    assert len(inp.shape) == 3, "Image should be h,w,3 "
    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes
    x = get_image_arr(inp, input_width, input_height, odering='channels_last')
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
    img1 = np.array(pr, dtype=np.uint8)
    return img1


def railway_unet(frame, n_classes, limit_box_width_height, show=True):
    img = predict_my(model=h, inp=frame)
    color = {1: (255, 100, 0), 2: (0, 100, 255), 3: (5, 255, 0)}
    account_classes = {1: 0, 2: 0, 3: 0}

    for c in range(1, n_classes, 1):
        only_cat_hsv = cv2.inRange(img, c, c)
        contours, hierarchy = cv2.findContours(only_cat_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            box = cv2.boundingRect(cnt)
            if clinket_rail(box, limit_box_width_height):
                frame = cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color[c], 5)
                account_classes[c] += 1
    if show:
        cv2.imshow('h1', frame)
        print(account_classes)
        cv2.waitKey(0)
    return f'Mixed: {account_classes[1]}, Dense: {account_classes[1]}, Diffuse: {account_classes[1]}.'


def clinket_rail(box, limit):
    return box[2] > limit[0] and box[3] > limit[1]


def net(n_classes, height, width, train, epochs):  # create a neural network for the segmentation of number
    img_input = Input(shape=(height, width, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    up0 = concatenate([UpSampling2D((2, 2))(conv4), conv3], axis=-1)
    conv41 = Conv2D(128, (3, 3), activation='relu', padding='same')(up0)
    conv41 = Dropout(0.2)(conv41)
    conv41 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv41)
    up1 = concatenate([UpSampling2D((2, 2))(conv41), conv2], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    out = Conv2D(n_classes, (1, 1), padding='same')(conv5)

    model = get_segmentation_model(img_input, out)  # this would build the segmentation model
    if train == True:
        model.train(
            train_images="images_prepped_train/",
            train_annotations="annotations_prepped_train/",
            checkpoints_path="content/", epochs=epochs)
        model.save('last.h5')

    return model


n_classes = 4
height = 768
width = 768

h = net(n_classes=n_classes, height=height, width=width, train=False, epochs=100)
h.load_weights('content/.10')

frame = cv2.imread('images_prepped_train/чашка 11.png')
frame = cv2.imread('чашка 4.jpg')
frame = cv2.resize(frame, (width, height))
limit_box_width_height = [10, 10]
account_classes = railway_unet(frame, n_classes, limit_box_width_height)
