import numpy as np
import os
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
# from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import decode_predictions
# from imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from keras.layers import merge, Input, Add, UpSampling2D
from keras.models import Model, load_model
from keras.utils import np_utils, to_categorical
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pickle
import keras.backend as K
import tensorflow as tf

def convert_to_color_segmentation(arr_2d, palette):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    R = np.zeros((arr_2d.shape[0], arr_2d.shape[1]), dtype=np.uint8)
    G = np.zeros((arr_2d.shape[0], arr_2d.shape[1]), dtype=np.uint8)
    B = np.zeros((arr_2d.shape[0], arr_2d.shape[1]), dtype=np.uint8)

    index_values = np.unique(arr_2d)
    for c, i in palette.items():
        if i in index_values:
            mask = arr_2d == i
            R[mask] = c[0]
            G[mask] = c[1]
            B[mask] = c[2]
            arr_3d = np.stack((R, G, B), 2)
            arr_3d = image.array_to_img(arr_3d)
    return arr_3d


def to_normal_tensor(x):
    y = x.argmax(axis=2)
    return y


def save_data(filepath, data):
    output = open(filepath, 'wb')
    pickle.dump(data, output, protocol=4)
    output.close()


def load_data(filepath):
    pkl_file = open(filepath, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data

def mean_IoU(y_true, y_pred):
    s = K.shape(y_true)

    # reshape such that w and h dim are multiplied together
    y_true_reshaped = K.reshape(y_true, tf.stack([-1, s[1] * s[2], s[-1]]))
    y_pred_reshaped = K.reshape(y_pred, tf.stack([-1, s[1] * s[2], s[-1]]))

    # correctly classified
    clf_pred = K.one_hot(K.argmax(y_pred_reshaped), num_classes=s[-1])
    equal_entries = K.cast(K.equal(clf_pred, y_true_reshaped), dtype='float32') * y_true_reshaped

    intersection = K.sum(equal_entries, axis=1)
    union_per_class = K.sum(y_true_reshaped, axis=1) + K.sum(y_pred_reshaped, axis=1)

    iou = intersection / (union_per_class - intersection)
    iou_mask = tf.is_finite(iou)
    iou_masked = tf.boolean_mask(iou, iou_mask)

    return K.mean(iou_masked)

root = os.getcwd()
'''
filepath_32 = root + "/model_fcn_32s_tran.hdf5"
FCN_32 = load_model(filepath_32)


filepath_16 = root + "/model_fcn_16s_tran.hdf5"
FCN_16 = load_model(filepath_16)
'''

filepath_8 = root + "/model_fcn_8s_tran_init.h5"
FCN_8 = load_model(filepath_8, custom_objects={"mean_IoU":mean_IoU})

# load saved image data in X_fin_test and y_fin_test
X_fin_test = load_data("X_fin_test.pkl")
y_fin_test = load_data("y_fin_test.pkl")

classes = {'background': 1, 'aeroplane': 2, 'bicycle': 3, 'bird': 4, 'boat': 5,
           'bottle': 6, 'bus': 7, 'car': 8, 'cat': 9,
           'chair': 10, 'cow': 11, 'diningtable': 12, 'dog': 13,
           'horse': 14, 'motorbike': 15, 'person': 16, 'potted-plant': 17,
           'sheep': 18, 'sofa': 19, 'train': 20, 'tv/monitor': 21}

palette = {
           (0, 0, 0): 1,
           (128, 0, 0): 2,
           (0, 128, 0): 3,
           (128, 128, 0): 4,
           (0, 0, 128): 5,
           (128, 0, 128): 6,
           (0, 128, 128): 7,
           (128, 128, 128): 8,
           (64, 0, 0): 9,
           (192, 0, 0): 10,
           (64, 128, 0): 11,
           (192, 128, 0): 12,
           (64, 0, 128): 13,
           (192, 0, 128): 14,
           (64, 128, 128): 15,
           (192, 128, 128): 16,
           (0, 64, 0): 17,
           (128, 64, 0): 18,
           (0, 192, 0): 19,
           (128, 192, 0): 20,
           (0, 64, 128): 21}

'''
predictions = FCN_32.predict(X_fin_test, batch_size=1, verbose=1)
print("FCN32 predictions: ")
for i in range(0,5):
    print(i)
    img_with_class = to_normal_tensor(predictions[i,:,:,:])
    img_with_rgb = convert_to_color_segmentation(img_with_class, palette)
    img_with_rgb
    y_fin_test = to_normal_tensor(y_fin_test)
    y_fin_test = convert_to_color_segmentation(y_fin_test)
    y_fin_test

predictions = FCN_16.predict(X_fin_test, batch_size=1, verbose=1)
print("FCN16 predictions: ")
for i in range(0,5):
    print(i)
    img_with_class = to_normal_tensor(predictions[i,:,:,:])
    img_with_rgb = convert_to_color_segmentation(img_with_class, palette)
    img_with_rgb
    y_fin_test = to_normal_tensor(y_fin_test)
    y_fin_test = convert_to_color_segmentation(y_fin_test)
    y_fin_test
'''
predictions = FCN_8.predict(X_fin_test, batch_size=1, verbose=1)
print("FCN8 predictions: ")
for i in range(0, 5):
    print(i)
    img_with_class = to_normal_tensor(predictions[i, :, :, :])
    img_with_rgb = convert_to_color_segmentation(img_with_class, palette)
    img_with_rgb.show()
    y_fin_test_image = to_normal_tensor(y_fin_test[i,:,:,:])
    y_fin_test_image = convert_to_color_segmentation(y_fin_test_image, palette)
    y_fin_test_image.show()
