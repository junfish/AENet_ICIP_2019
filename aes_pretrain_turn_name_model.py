from keras import Input
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.applications.resnet50 import ResNet50
from keras.layers.convolutional import ZeroPadding2D, Conv2D, MaxPooling2D
from keras import layers
from keras.layers import AveragePooling2D, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import concatenate
from keras import regularizers
from keras.models import Model
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import random
import h5py
import os

def conv_block(input_tensor, kernel_size, filters, stage, block, strides = (2, 2), social = None):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch_' + social
    bn_name_base = 'bn' + str(stage) + block + '_branch_' + social
    prelu_name_base = 'prelu' + str(stage) + block + '_branch_' + social
    x = Conv2D(filters1, (1, 1), strides = strides, name = conv_name_base + '_2a')(input_tensor)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '_2a')(x)
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    x = PReLU(name = prelu_name_base + '_2a')(x)

    x = Conv2D(filters2, kernel_size, padding = 'same', name = conv_name_base + '_2b')(x)
    x = BatchNormalization(axis=bn_axis, name = bn_name_base + '_2b')(x)
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    x = PReLU(name = prelu_name_base + '_2b')(x)

    x = Conv2D(filters3, (1, 1), name = conv_name_base + '_2c')(x)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '_2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides = strides, name = conv_name_base + '_1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '_1')(shortcut)

    x = layers.add([x, shortcut])
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    x = PReLU(name = prelu_name_base + '_2c')(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block, social):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch_' + social
    bn_name_base = 'bn' + str(stage) + block + '_branch_' + social
    prelu_name_base = 'prelu' + str(stage) + block + '_branch_' + social
    x = Conv2D(filters1, (1, 1), name = conv_name_base + '_2a')(input_tensor)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '_2a')(x)
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    x = PReLU(name = prelu_name_base + "_2a")(x)

    x = Conv2D(filters2, kernel_size, padding = 'same', name = conv_name_base + '_2b')(x)
    x = BatchNormalization(axis = bn_axis, name=bn_name_base + '_2b')(x)
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    x = PReLU(name = prelu_name_base + "_2b")(x)

    x = Conv2D(filters3, (1, 1), name = conv_name_base + '_2c')(x)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '_2c')(x)

    x = layers.add([x, input_tensor])
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    x = PReLU(name = prelu_name_base + "_2c")(x)
    return x

def emo_resnet50(Aes_input_tensor):
    # input_shape = (224, 224, 3,)
    # img_input_1 = Input(shape=input_shape)  # Emotions
    # img_input_2 = Input(shape=input_shape)  # Aesthetics

    # if K.image_data_format() == 'channels_last':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1
    bn_axis = 3

    # x_1 = ZeroPadding2D(padding=(3, 3), name='conv1_pad_x_1')(Emo_input_tensor)
    # x_1 = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1_x_1')(x_1)
    # x_1 = BatchNormalization(axis=bn_axis, name='bn_conv1_x_1')(x_1)
    # x_1 = Activation('relu')(x_1)
    # x_1 = MaxPooling2D((3, 3), strides=(2, 2))(x_1)

    x_2 = ZeroPadding2D(padding=(3, 3), name='conv1_pad_x_2')(Aes_input_tensor)
    x_2 = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1_x_2')(x_2)
    x_2 = BatchNormalization(axis=bn_axis, name='bn_conv1_x_2')(x_2)
    # x_2 = Activation('relu')(x_2)
    # x_2 = LeakyReLU(alpha=0.01)(x_2)
    x_2 = PReLU(name = "prelu_x_2")(x_2)
    x_2 = MaxPooling2D((3, 3), strides=(2, 2))(x_2)

    # x_1, x_2 = merge(x_1, x_2, "1", 64)

    # x_1 = conv_block(x_1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), social='x_1')
    # x_1 = identity_block(x_1, 3, [64, 64, 256], stage=2, block='b', social="x_1")
    # x_1 = identity_block(x_1, 3, [64, 64, 256], stage=2, block='c', social="x_1")

    x_2 = conv_block(x_2, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), social='x_2')
    x_2 = identity_block(x_2, 3, [64, 64, 256], stage=2, block='b', social="x_2")
    x_2 = identity_block(x_2, 3, [64, 64, 256], stage=2, block='c', social="x_2")

    # x_1, x_2 = merge(x_1, x_2, '2', 256)

    # x_1 = conv_block(x_1, 3, [128, 128, 512], stage=3, block='a', social='x_1')
    # x_1 = identity_block(x_1, 3, [128, 128, 512], stage=3, block='b', social='x_1')
    # x_1 = identity_block(x_1, 3, [128, 128, 512], stage=3, block='c', social='x_1')
    # x_1 = identity_block(x_1, 3, [128, 128, 512], stage=3, block='d', social='x_1')

    x_2 = conv_block(x_2, 3, [128, 128, 512], stage=3, block='a', social="x_2")
    x_2 = identity_block(x_2, 3, [128, 128, 512], stage=3, block='b', social="x_2")
    x_2 = identity_block(x_2, 3, [128, 128, 512], stage=3, block='c', social="x_2")
    x_2 = identity_block(x_2, 3, [128, 128, 512], stage=3, block='d', social="x_2")

    # x_1, x_2 = merge(x_1, x_2, '3', 512)
    #
    # x_1 = conv_block(x_1, 3, [256, 256, 1024], stage=4, block='a', social='x_1')
    # x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='b', social='x_1')
    # x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='c', social='x_1')
    # x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='d', social='x_1')
    # x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='e', social='x_1')
    # x_1 = identity_block(x_1, 3, [256, 256, 1024], stage=4, block='f', social='x_1')

    x_2 = conv_block(x_2, 3, [256, 256, 1024], stage=4, block='a', social='x_2')
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='b', social='x_2')
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='c', social='x_2')
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='d', social='x_2')
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='e', social='x_2')
    x_2 = identity_block(x_2, 3, [256, 256, 1024], stage=4, block='f', social='x_2')

    # x_1, x_2 = merge(x_1, x_2, '4', 1024)
    #
    # x_1 = conv_block(x_1, 3, [512, 512, 2048], stage=5, block='a', social='x_1')
    # x_1 = identity_block(x_1, 3, [512, 512, 2048], stage=5, block='b', social='x_1')
    # x_1 = identity_block(x_1, 3, [512, 512, 2048], stage=5, block='c', social='x_1')

    x_2 = conv_block(x_2, 3, [512, 512, 2048], stage=5, block='a', social='x_2')
    x_2 = identity_block(x_2, 3, [512, 512, 2048], stage=5, block='b', social='x_2')
    x_2 = identity_block(x_2, 3, [512, 512, 2048], stage=5, block='c', social='x_2')

    # x_1, x_2 = merge(x_1, x_2, '5', 2048)

    # x_1 = AveragePooling2D((7, 7), name='avg_pool_x_1')(x_1)
    # x_1 = Flatten(name='flatten_x_1')(x_1)
    # # x_1 = Dense(1000, activation='relu', name='fc_x_1')(x_1)
    # x_1 = Dense(2, activation='softmax', name='predictions_x_1')(x_1)

    x_2 = AveragePooling2D((7, 7), name='avg_pool_x_2')(x_2)
    x_2 = Flatten(name='flatten_x_2')(x_2)
    # x_2 = Dense(1000, activation='relu', name='fc_x_2')(x_2)
    x_2 = Dense(2, activation='softmax', name='predictions_x_2')(x_2)

    model = Model(inputs = Aes_input_tensor, outputs = x_2)
    return model

def accuracy(predict_list, truth_list):
    all = len(truth_list)
    acc = 0
    for (predict_list_element, truth_list_element) in zip(predict_list, truth_list):
        if predict_list_element.index(max(predict_list_element)) == truth_list_element.index(max(truth_list_element)):
            acc += 1
    return (float(acc) / float(all))

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    input_shape = (224, 224, 3,)
    aes_input = Input(shape=input_shape, name='aes_input')
    model = emo_resnet50(Aes_input_tensor=aes_input)  # weights = "imagenet"
    model.summary()
    # x = Flatten(name="flatten_x_1")(out_part_model)
    # img_output = Dense(8, activation='softmax', name='predictions_x_1')(x)
    # emo_model = Model(inputs=emo_input, outputs=img_output, name="emo_model")
    # model.load_weights("/home/yujun/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)
    model.load_weights("/home/yujun/PycharmProjects/NEW_WORK/weights/aes4_train_Prelu_dropout_zero_centre_lr_2e-4.h5")
    # f1 = h5py.File("/home/yujun/PycharmProjects/Emotion_Recognition/aes/aes_256.h5", "r")
    # emo_X_train = f1["train_images"][:]
    # emo_Y_train = f1["train_labels"][:]
    # emo_X_test = f1["test_images"][:]
    # emo_Y_test = f1["test_labels"][:]
    # f3 = h5py.File("/home/yujun/PycharmProjects/NEW_WORK/ava/test_data.hdf5")
    # aes_X_test = f3["images"][:]
    # aes_test_labels = f3["labels"][:]
    # f3.close()
    # sss = np.mean(aes_X_test, axis=0)
    # aes_train_images = aes_X_test - sss
    # aes_Y_test_list = []
    # for label in aes_test_labels:
    #     if int(label) == 0:
    #         aes_Y_test_list.append(np.array([1, 0]))
    #     else:
    #         aes_Y_test_list.append(np.array([0, 1]))
    # aes_Y_test = np.array(aes_Y_test_list)
    f2 = h5py.File("/home/yujun/PycharmProjects/NEW_WORK/AVA/h5_sets/aes_2class_5percent_zero_centre_float32.h5")
    # f3 = h5py.File("/home/yujun/PycharmProjects/Emotion_Recognition/ava/test_data.hdf5")
    aes_X_test = f2["images_test"][:]
    aes_Y_test = f2["labels_test"][:]
    f2.close()
    # test_batches = len(aes_X_test) / 32 + 1
    score = model.predict(aes_X_test[:, 0:224, 0:224, :], verbose=1)
    print accuracy(score.tolist(), aes_Y_test.tolist())
    model.save_weights('/home/yujun/PycharmProjects/NEW_WORK/weights/aes_relu_newname_transfer_lr-4.h5')