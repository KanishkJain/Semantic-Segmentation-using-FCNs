import numpy as np
import os
import time
import keras.backend as K
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
# from vgg16 import VGG16
from keras.preprocessing import image
#from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import decode_predictions, preprocess_input
# from imagenet_utils import decode_predictions
from keras.layers import Dense, Lambda, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from keras.layers import merge, Input, Add, UpSampling2D
from keras.models import Model
from keras.utils import np_utils, to_categorical
from keras.initializers import RandomNormal
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pickle
from keras import regularizers

# img_path = 'test.png'
# image.load_img('test.png', target_size=(224, 224))
# test = image.img_to_array(test_img)
# print (test.shape)
# test = np.expand_dims(test, axis=0)
# print (test.shape)
# test = preprocess_input(test)
# print('Input image shape:', test.shape)


classes = {'background': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4,
           'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8,
           'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12,
           'horse': 13, 'motorbike': 14, 'person': 15, 'potted-plant': 16,
           'sheep': 17, 'sofa': 18, 'train': 19, 'tv/monitor': 20}

palette = {
           (0, 0, 0): 0,
           (128, 0, 0): 1,
           (0, 128, 0): 2,
           (128, 128, 0): 3,
           (0, 0, 128): 4,
           (128, 0, 128): 5,
           (0, 128, 128): 6,
           (128, 128, 128): 7,
           (64, 0, 0): 8,
           (192, 0, 0): 9,
           (64, 128, 0): 10,
           (192, 128, 0): 11,
           (64, 0, 128): 12,
           (192, 0, 128): 13,
           (64, 128, 128): 14,
           (192, 128, 128): 15,
           (0, 64, 0): 16,
           (128, 64, 0): 17,
           (0, 192, 0): 18,
           (128, 192, 0): 19,
           (0, 64, 128): 20}


def convert_from_color_segmentation(arr_3d, palette):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


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


def to_categorical_tensor(x3d, n_cls):
    batch_size, n_rows, n_cols = x3d.shape
    x1d = x3d.ravel()
    y1d = to_categorical(x1d, num_classes=n_cls)
    y4d = y1d.reshape([batch_size, n_rows, n_cols, n_cls])
    return y4d


def to_normal_tensor(x):
    y = x.argmax(axis=2)
    return y


''' 
def save_data(filepath, data):
    with open(filepath, 'w') as outfile:
        for data_slice in data:
            np.savetxt(outfile, data_slice, fmt='%-7.2f')
'''


def save_data(filepath, data):
    output = open(filepath, 'wb')
    pickle.dump(data, output, protocol=4)
    output.close()


def load_data(filepath):
    pkl_file = open(filepath, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1] + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(
        tf.to_float(legal_labels))

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

def IoU(y_true, y_pred):

    y_true = y_true.eval()
    y_pred = y_pred.eval()

    y_true = y_true.argmax(axis=2)
    y_pred = y_pred.argmax(axis=2)

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    cm = confusion_matrix(y_true, y_pred)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    mean_iu = np.mean(TP / (TP + FP + FN))
    return mean_iu

def depth_softmax(matrix):
    sigmoid = lambda x: 1 / (1 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
    return softmax_matrix

# Loading train data in python
root = os.getcwd()
y_path = root + "/SegmentationClass/"
X_path = root + "/JPEGImages/"
txt_file = "train.txt"

y_ext = '.png'
X_ext = '.jpg'

if os.path.isfile("X_train.pkl") == False:
    y_train_matrix = []
    X_train_matrix = []

    with open(txt_file, 'rb') as f:

        for img_name in f:
            img_base_name = img_name.strip().decode("utf-8")
            y_img_name = os.path.join(y_path, str(img_base_name)) + y_ext
            y_img_file = image.load_img(y_img_name, target_size=(224, 224))
            y_img = image.img_to_array(y_img_file)

            X_img_name = os.path.join(X_path, str(img_base_name)) + X_ext
            X_img_file = image.load_img(X_img_name, target_size=(224, 224))
            X_img = image.img_to_array(X_img_file)
            X_img = preprocess_input(X_img)

            if (len(y_img.shape) > 2):
                y_img = convert_from_color_segmentation(y_img, palette)
                y_train_matrix.append(y_img)
                X_train_matrix.append(X_img)

    y_train_matrix = np.asarray(y_train_matrix)
    y_train = to_categorical_tensor(y_train_matrix, 21)
    y_train.shape

    X_train = np.asarray(X_train_matrix)
    X_train.shape

    save_data("X_train.pkl", X_train)
    save_data("y_train.pkl", y_train)

else:
    X_train = load_data("X_train.pkl")
    y_train = load_data("y_train.pkl")

root = os.getcwd()
y_path = root + "/SegmentationClass/"
X_path = root + "/JPEGImages/"
txt_file = "val.txt"
# path_converted = "F:/Image Detection/ConvSeg/"

y_ext = '.png'
X_ext = '.jpg'

# if not os.path.isdir(path_converted):
#  os.makedirs(path_converted)

count = 0
y_test_matrix = []
X_test_matrix = []

if os.path.isfile("X_fin_test.pkl") == False:
    with open(txt_file, 'rb') as f:

        for img_name in f:
            img_base_name = img_name.strip().decode("utf-8")
            y_img_name = os.path.join(y_path, str(img_base_name)) + y_ext
            y_img_file = image.load_img(y_img_name, target_size=(224, 224))
            y_img = image.img_to_array(y_img_file)

            # img = imread(img_name)

            X_img_name = os.path.join(X_path, str(img_base_name)) + X_ext
            X_img_file = image.load_img(X_img_name, target_size=(224, 224))
            X_img = image.img_to_array(X_img_file)
            X_img = preprocess_input(X_img)

            if (len(y_img.shape) > 2):
                y_img = convert_from_color_segmentation(y_img, palette)
                # print(np.unique(img))
                y_test_matrix.append(y_img)
                X_test_matrix.append(X_img)

    y_test_matrix = np.asarray(y_test_matrix)
    y_test = to_categorical_tensor(y_test_matrix, 21)
    y_test.shape

    X_test = np.asarray(X_test_matrix)
    X_test.shape

    # splitting X_test into Final_Test and Val data
    X_val, X_fin_test, y_val, y_fin_test = train_test_split(X_test, y_test, test_size=0.5, random_state=123)

    save_data("X_val.pkl", X_val)
    save_data("y_val.pkl", y_val)
    save_data("X_fin_test.pkl", X_fin_test)
    save_data("y_fin_test.pkl", y_fin_test)

else:
    X_val = load_data("X_val.pkl")
    y_val = load_data("y_val.pkl")
    X_fin_test = load_data("X_fin_test.pkl")
    y_fin_test = load_data("y_fin_test.pkl")

############# model definition #################

image_input = Input(shape=(224, 224, 3))

# FCN32s
model_32 = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')

last_layer = model_32.get_layer('block5_pool').output
x = last_layer

# Convolutional layers transfered from fully-connected layers
x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(x)
x = Dropout(0.5, name='dropout1')(x)
x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(x)
x = Dropout(0.5, name='dropout2')(x)

# classifying layer
x = Conv2D(21, (1, 1), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None), kernel_regularizer=regularizers.l2(0.01), activation='linear', padding='valid', strides=(1, 1),
           name='final_conv_32')(x)

x = Conv2DTranspose(21, (3, 3), strides=(32, 32), padding='same', name='upsampling32', kernel_regularizer=regularizers.l2(0.01))(x)
model_32 = Model(image_input, x)
model_32.summary()

# FCN16s
# model_16 = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

second_last_layer = model_32.get_layer('block4_pool').output
x = Conv2D(21, (1, 1), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None), activation='linear', padding='valid', strides=(1, 1),
           name='final_conv_16', kernel_regularizer=regularizers.l2(0.01))(second_last_layer)

y = model_32.get_layer('final_conv_32').output
y = Conv2DTranspose(21, (3, 3), strides=(2, 2), padding='same', name='upsampling2x_16', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(y)
# new_layer = merge[x, new_layer, mode = 'sum']
new_layer = Add()([x, y])
new_layer = Conv2DTranspose(21, (3, 3), strides=(16, 16), padding='same', name='upsampling16', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(new_layer)

model_16 = Model(image_input, new_layer)
model_16.summary()

# FCN8s

second_last_layer = model_32.get_layer('block4_pool').output
x = Conv2D(21, (1, 1), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None), activation='linear', padding='valid', strides=(1, 1),
           name='final_conv_8_1')(second_last_layer)
x = Conv2DTranspose(21, (3, 3), strides=(2, 2), padding='same', name='upsampling2x_8s', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(x)

third_last_layer = model_32.get_layer('block3_pool').output
y = Conv2D(21, (1, 1), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None), activation='linear', padding='valid', strides=(1, 1),
           name='final_conv_8_2')(third_last_layer)

z = model_32.get_layer('final_conv_32').output
z = Conv2DTranspose(21, (3, 3), strides=(4, 4), padding='same', name='upsampling4x_8s', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(z)

new_layer = Add()([x, y, z])
new_layer = Conv2DTranspose(21, (3, 3), strides=(8, 8), padding='same', name='upsampling8s', activation = "softmax", kernel_regularizer=regularizers.l2(0.01), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(new_layer)
    
model_8 = Model(image_input, new_layer)
model_8.summary()

# Using Resnet50 instead of VGG

# model_res = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')

# x = model_res.get_layer('activation_49').output
# x = Dropout(0.5)(x)
# x = Conv2D(21, (1,1), name = 'pred_32',init='zero', padding = 'same')(x)

# x = Conv2DTranspose(21, (3,3), strides = (32,32), padding = 'same', name = 'upsampling_res')(new_layer)


# 32 layers


# Training all the algos

# freeze all the layers except the dense layers
for layer in model_32.layers[:-6]:
    layer.trainable = False

# freeze all the layers except the dense layers
for layer in model_16.layers[:-9]:
    layer.trainable = False

# freeze all the layers except the dense layers
for layer in model_8.layers[:-11]:
    layer.trainable = False

sgd = optimizers.SGD(lr=10e-4, decay=2e-5, momentum=0.9, nesterov=True)

'''
#model_32.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
model_32.compile(loss = softmax_sparse_crossentropy_ignoring_last_label, optimizer = sgd, metrics = [sparse_accuracy_ignoring_last_label])



t=time.time()
#t = now()

filepath="model_fcn_32s_tran.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model_32.fit(X_train, y_train, batch_size=1, epochs=5, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks_list)
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = model_32.evaluate(X_fin_test, y_fin_test, batch_size=1, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('model_32_acc_tran.png', bbox_inches='tight')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('model_32_loss_tran.png', bbox_inches='tight')


model_16.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

t=time.time()
#t = now()

filepath="model_fcn_16s_tran.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model_16.fit(X_train, y_train, batch_size=1, epochs=5, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks_list)
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = model_16.evaluate(X_fin_test, y_fin_test, batch_size=1, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('model_16_acc_up_tran.png', bbox_inches='tight')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('model_16_loss_up_tran.png', bbox_inches='tight')
'''

model_8.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[mean_IoU, 'accuracy'])

t = time.time()
# t = now()
'''
class_weight = {0: 1.,
                1: 20.,
                2: 20.,
                3: 20.,
                4: 20.,
                5: 20.,
                6: 20.,
                7: 20.,
                8: 20.,
                9: 20.,
                10: 20.,
                11: 20.,
                12: 20.,
                13: 20.,
                14: 20.,
                15: 20.,
                16: 20.,
                17: 20.,
                18: 20.,
                19: 20.,
                20: 20.,
                }
'''
filepath = "model_fcn_8s_tran_init.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model_8.fit(X_train, y_train, batch_size=1, epochs=1, verbose=1, validation_data=(X_val, y_val),
                      callbacks=callbacks_list)
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = model_8.evaluate(X_fin_test, y_fin_test, batch_size=1, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

'''
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('model_8_acc_tran.png', bbox_inches='tight')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('model_8_loss_tran.png', bbox_inches='tight')
'''
# model_32.save("model_fcn32s_fin_tran.h5")
# model_16.save("model_fcn16s_fin_tran.h5")
model_8.save("model_fcn8s_fin_tran_init.h5")
