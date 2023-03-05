from keras import Input
import tensorflow as tf
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
from resnet50_block import block1, block2, block3, block4, block5
from NDDR_initializers import NDDR_init
import keras
from Emo_NDDR import Emo_NDDR_resnet
from Aes_NDDR import Aes_NDDR_resnet
import os
import gc
def for_generate(X, Y, batch_size):
    while 1:
        cnt = 0
        x_batch =[]
        y_batch =[]
        for item in np.arange(len(Y)):
            index = item
            start_x_1 = random.randint(0, 31)
            start_y_1 = random.randint(0, 31)
            x_batch.append(X[index][start_x_1:start_x_1 + 224, start_y_1:start_y_1 + 224])
            y_batch.append(Y[index])
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield (np.array(x_batch), np.array(y_batch))
                x_batch = []
                y_batch = []

def data_split(X, Y, test_rate):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_rate)
    return X_train, X_test, Y_train, Y_test

def accuracy(predict_list, truth_list):
    all = len(truth_list)
    acc = 0
    for (predict_list_element, truth_list_element) in zip(predict_list, truth_list):
        if predict_list_element.index(max(predict_list_element)) == truth_list_element.index(max(truth_list_element)):
            acc += 1
    return (float(acc) / float(all))

def kld_mean(truth_distribution, predict_distribution):
    sum = 0.0
    for (truth, predict) in zip(truth_distribution, predict_distribution):
        for (x, y) in zip (truth, predict):
            if (str(x * np.log(x / y)) != "inf") & (str(x * np.log(x / y)) != "nan"):
                sum += x * np.log(x / y)
    return sum / len(truth_distribution)

def conv_block(input_tensor, kernel_size, filters, stage, block, strides = (2, 2), social = None):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch_' + social
    bn_name_base = 'bn' + str(stage) + block + '_branch_' + social
    relu_name_base = 'relu' + str(stage) + block + '_branch_' + social
    x = Conv2D(filters1, (1, 1), strides = strides, name = conv_name_base + '_2a')(input_tensor)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '_2a')(x)
    x = Activation('relu', name = relu_name_base + '_2a')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU(name = prelu_name_base + '_2a')(x)

    x = Conv2D(filters2, kernel_size, padding = 'same', name = conv_name_base + '_2b')(x)
    x = BatchNormalization(axis=bn_axis, name = bn_name_base + '_2b')(x)
    x = Activation('relu', name = relu_name_base + '_2b')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU(name = prelu_name_base + '_2b')(x)

    x = Conv2D(filters3, (1, 1), name = conv_name_base + '_2c')(x)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '_2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides = strides, name = conv_name_base + '_1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '_1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu', name = relu_name_base + '_2c')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU(name = prelu_name_base + '_2c')(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block, social):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch_' + social
    bn_name_base = 'bn' + str(stage) + block + '_branch_' + social
    relu_name_base = 'relu' + str(stage) + block + '_branch_' + social
    x = Conv2D(filters1, (1, 1), name = conv_name_base + '_2a')(input_tensor)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '_2a')(x)
    x = Activation('relu', name = relu_name_base + "_2a")(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU(name = prelu_name_base + "_2a")(x)

    x = Conv2D(filters2, kernel_size, padding = 'same', name = conv_name_base + '_2b')(x)
    x = BatchNormalization(axis = bn_axis, name=bn_name_base + '_2b')(x)
    x = Activation('relu', name = relu_name_base + "_2b")(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU(name = prelu_name_base + "_2b")(x)

    x = Conv2D(filters3, (1, 1), name = conv_name_base + '_2c')(x)
    x = BatchNormalization(axis = bn_axis, name = bn_name_base + '_2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu', name = relu_name_base + "_2c")(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU(name = prelu_name_base + "_2c")(x)
    return x

def aes_merge(x_1, x_2, n, filters):
    name_1 = 'Aes_NDDR_' + n
    name_2 = 'Aes_NDDR_' + n + '_share'
    pool_merge = concatenate([x_1, x_2])
    x_1 = Conv2D(filters, (1, 1), strides=(1, 1), activation='relu', trainable = False, kernel_initializer=NDDR_init(parameter_1=0.5,parameter_2=0.5), name=name_1)(pool_merge)
    x_2 = Conv2D(filters, (1, 1), strides=(1, 1), activation='relu', trainable = False, kernel_initializer=NDDR_init(parameter_1=0.5,parameter_2=0.5), name=name_2)(pool_merge)
    # x_2 = Conv2D(filters, (1, 1), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.01),
    #              kernel_initializer=initializers.glorot_normal(), name=name_2)(pool_merge)
    return x_1, x_2

def emo_merge(x_1, x_2, n, filters):
    name_1 = 'Emo_NDDR_' + n
    name_2 = 'Emo_NDDR_' + n + '_share'
    pool_merge = concatenate([x_1, x_2])
    x_1 = Conv2D(filters, (1, 1), strides=(1, 1), activation='relu', trainable = False, kernel_initializer=NDDR_init(parameter_1=0.5,parameter_2=0.5), name=name_1)(pool_merge)
    x_2 = Conv2D(filters, (1, 1), strides=(1, 1), activation='relu', trainable = False, kernel_initializer=NDDR_init(parameter_1=0.5,parameter_2=0.5), name=name_2)(pool_merge)
    return x_1, x_2

def aes_merge_1(x_1, x_2, n, filters):
    name = 'Aes_NDDR_' + n
    # pool_bn_x_1 = BatchNormalization()(x_1)
    # pool_bn_x_2 = BatchNormalization()(x_2)
    pool_merge = concatenate([x_1, x_2])
    x_1 = Conv2D(filters, (1, 1), strides = (1,1), activation = 'relu', trainable = False, kernel_initializer=NDDR_init(parameter_1=0.5,parameter_2=0.5), name = name)(pool_merge)
    return x_1

def emo_merge_1(x_1, x_2, n, filters):
    name = 'Emo_NDDR_' + n
    # pool_merge = concatenate([x_1, x_2])
    # pool_bn = BatchNormalization()(pool_merge)
    # x_1 = Conv2D(filters, (1, 1), strides = (1,1), activation = 'relu', kernel_regularizer = regularizers.l2(0.01), name = name)(pool_bn)
    # pool_bn_x_1 = BatchNormalization()(x_1)
    # pool_bn_x_2 = BatchNormalization()(x_2)
    pool_merge = concatenate([x_1, x_2])
    x_1 = Conv2D(filters, (1, 1), strides = (1,1), activation = 'relu', trainable = False, kernel_initializer=NDDR_init(parameter_1=0.5,parameter_2=0.5), name = name)(pool_merge)
    return x_1

def aes_block1(aes_input):

    bn_axis = 3

    # aes_block1_model = Model(inputs=base_model.input, outputs=base_model.get_layer("max_pooling2d_1").output)
    aes = ZeroPadding2D(padding=(3, 3), name='conv1_pad_aes')(aes_input)
    aes = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1_aes')(aes)
    aes = BatchNormalization(axis=bn_axis, name='bn_conv1_aes')(aes)
    aes = Activation('relu', name="activation_aes_1")(aes)
    aes = MaxPooling2D((3, 3), strides=(2, 2), name="max_pooling2d_aes_1")(aes)
    aes_block1_model = Model(inputs=aes_input, outputs=aes, name = "aes_block1")

    return aes_block1_model

def share_block1(share_input):

    bn_axis = 3

    share = ZeroPadding2D(padding=(3, 3), name='conv1_pad_share')(share_input)
    share = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1_share')(share)
    share = BatchNormalization(axis=bn_axis, name='bn_conv1_share')(share)
    share = Activation('relu', name = "activation_share_1")(share)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU()(x)
    share = MaxPooling2D((3, 3), strides=(2, 2), name = "max_pooling2d_aes_1")(share)

    share_block1_model = Model(inputs=share_input, outputs=share, name = "share_block1")
    return share_block1_model

def emo_block1(emo_input):
    bn_axis = 3

    # aes_block1_model = Model(inputs=base_model.input, outputs=base_model.get_layer("max_pooling2d_1").output)
    emo = ZeroPadding2D(padding=(3, 3), name='conv1_pad_emo')(emo_input)
    emo = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1_emo')(emo)
    emo = BatchNormalization(axis=bn_axis, name='bn_conv1_emo')(emo)
    emo = Activation('relu', name="activation_emo_1")(emo)
    emo = MaxPooling2D((3, 3), strides=(2, 2), name="max_pooling2d_emo_1")(emo)
    emo_block1_model = Model(inputs=emo_input, outputs=emo, name = "emo_block1")

    return emo_block1_model


def aes_block2(aes_input):

    aes = conv_block(aes_input, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), social = "aes")
    aes = identity_block(aes, 3, [64, 64, 256], stage=2, block='b', social = "aes")
    aes = identity_block(aes, 3, [64, 64, 256], stage=2, block='c', social = "aes")
    aes_block2_model = Model(inputs=aes_input, outputs=aes, name = "aes_block2")
    return aes_block2_model

def share_block2(share_input):

    share = conv_block(share_input, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), social="share")
    share = identity_block(share, 3, [64, 64, 256], stage=2, block='b', social="share")
    share = identity_block(share, 3, [64, 64, 256], stage=2, block='c', social="share")
    share_block2_model = Model(inputs=share_input, outputs=share, name = "share_block2")

    return share_block2_model

def emo_block2(emo_input):

    emo = conv_block(emo_input, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), social="emo")
    emo = identity_block(emo, 3, [64, 64, 256], stage=2, block='b', social="emo")
    emo = identity_block(emo, 3, [64, 64, 256], stage=2, block='c', social="emo")
    emo_block2_model = Model(inputs=emo_input, outputs=emo, name = "emo_block2")

    return emo_block2_model


def aes_block3(aes_input):

    aes = conv_block(aes_input, 3, [128, 128, 512], stage=3, block='a', social="aes")
    aes = identity_block(aes, 3, [128, 128, 512], stage=3, block='b', social="aes")
    aes = identity_block(aes, 3, [128, 128, 512], stage=3, block='c', social="aes")
    aes = identity_block(aes, 3, [128, 128, 512], stage=3, block='d', social="aes")
    aes_block3_model = Model(inputs=aes_input, outputs=aes, name = "aes_block3")

    return aes_block3_model

def share_block3(share_input):

    share = conv_block(share_input, 3, [128, 128, 512], stage=3, block='a', social="share")
    share = identity_block(share, 3, [128, 128, 512], stage=3, block='b', social="share")
    share = identity_block(share, 3, [128, 128, 512], stage=3, block='c', social="share")
    share = identity_block(share, 3, [128, 128, 512], stage=3, block='d', social="share")
    share_block3_model = Model(inputs=share_input, outputs=share, name = "share_block3")

    return share_block3_model

def emo_block3(emo_input):

    emo = conv_block(emo_input, 3, [128, 128, 512], stage=3, block='a', social="emo")
    emo = identity_block(emo, 3, [128, 128, 512], stage=3, block='b', social="emo")
    emo = identity_block(emo, 3, [128, 128, 512], stage=3, block='c', social="emo")
    emo = identity_block(emo, 3, [128, 128, 512], stage=3, block='d', social="emo")
    emo_block3_model = Model(inputs=emo_input, outputs=emo, name = "emo_block3")

    return emo_block3_model


def aes_block4(aes_input):

    aes = conv_block(aes_input, 3, [256, 256, 1024], stage=4, block='a', social="aes")
    aes = identity_block(aes, 3, [256, 256, 1024], stage=4, block='b', social="aes")
    aes = identity_block(aes, 3, [256, 256, 1024], stage=4, block='c', social="aes")
    aes = identity_block(aes, 3, [256, 256, 1024], stage=4, block='d', social="aes")
    aes = identity_block(aes, 3, [256, 256, 1024], stage=4, block='e', social="aes")
    aes = identity_block(aes, 3, [256, 256, 1024], stage=4, block='f', social="aes")

    aes_block4_model = Model(inputs=aes_input, outputs=aes, name = "aes_block4")

    return aes_block4_model

def share_block4(share_input):

    share = conv_block(share_input, 3, [256, 256, 1024], stage=4, block='a', social="share")
    share = identity_block(share, 3, [256, 256, 1024], stage=4, block='b', social="share")
    share = identity_block(share, 3, [256, 256, 1024], stage=4, block='c', social="share")
    share = identity_block(share, 3, [256, 256, 1024], stage=4, block='d', social="share")
    share = identity_block(share, 3, [256, 256, 1024], stage=4, block='e', social="share")
    share = identity_block(share, 3, [256, 256, 1024], stage=4, block='f', social="share")

    share_block4_model = Model(inputs=share_input, outputs=share, name = "share_block4")

    return share_block4_model

def emo_block4(emo_input):

    emo = conv_block(emo_input, 3, [256, 256, 1024], stage=4, block='a', social="emo")
    emo = identity_block(emo, 3, [256, 256, 1024], stage=4, block='b', social="emo")
    emo = identity_block(emo, 3, [256, 256, 1024], stage=4, block='c', social="emo")
    emo = identity_block(emo, 3, [256, 256, 1024], stage=4, block='d', social="emo")
    emo = identity_block(emo, 3, [256, 256, 1024], stage=4, block='e', social="emo")
    emo = identity_block(emo, 3, [256, 256, 1024], stage=4, block='f', social="emo")

    emo_block4_model = Model(inputs=emo_input, outputs=emo, name = "emo_block4")

    return emo_block4_model


def aes_block5(aes_input):
    aes = conv_block(aes_input, 3, [512, 512, 2048], stage=5, block='a', social='aes')
    aes = identity_block(aes, 3, [512, 512, 2048], stage=5, block='b', social='aes')
    aes = identity_block(aes, 3, [512, 512, 2048], stage=5, block='c', social='aes')

    aes_block5_model = Model(inputs=aes_input, outputs=aes, name = "aes_block5")

    return aes_block5_model

def share_block5(share_input):

    share = conv_block(share_input, 3, [512, 512, 2048], stage=5, block='a', social='share')
    share = identity_block(share, 3, [512, 512, 2048], stage=5, block='b', social='share')
    share = identity_block(share, 3, [512, 512, 2048], stage=5, block='c', social='share')

    share_block5_model = Model(inputs=share_input, outputs=share, name = "share_block5")

    return share_block5_model

def emo_block5(emo_input):

    emo = conv_block(emo_input, 3, [512, 512, 2048], stage=5, block='a', social='emo')
    emo = identity_block(emo, 3, [512, 512, 2048], stage=5, block='b', social='emo')
    emo = identity_block(emo, 3, [512, 512, 2048], stage=5, block='c', social='emo')

    emo_block5_model = Model(inputs=emo_input, outputs=emo, name = "emo_block5")

    return emo_block5_model

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    ################################################  Modeling  #################################################
    input_shape = (224, 224, 3,)
    input = Input(shape=input_shape)
    emo_input = Input(shape=input_shape, name='emo_input')
    emo_block1_model = emo_block1(emo_input=input)
    value1 = emo_block1_model.get_weights()
    emo_block1_model.load_weights("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/block_weights/imagenet_block1.h5")
    value2 = emo_block1_model.get_weights()
    emo_output = emo_block1_model(emo_input)

    block2_shape = (55, 55, 64,)
    block2_input = Input(shape=block2_shape)
    emo_block2_model = emo_block2(emo_input=block2_input)
    value3 = emo_block2_model.get_weights()
    emo_block2_model.load_weights("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/block_weights/imagenet_block2.h5")
    value4 = emo_block2_model.get_weights()
    emo_output = emo_block2_model(emo_output)

    block3_shape = (55, 55, 256,)
    block3_input = Input(shape=block3_shape)
    emo_block3_model = emo_block3(emo_input=block3_input)
    value5 = emo_block3_model.get_weights()
    emo_block3_model.load_weights("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/block_weights/imagenet_block3.h5")
    value6 = emo_block3_model.get_weights()
    emo_output = emo_block3_model(emo_output)

    block4_shape = (28, 28, 512,)
    block4_input = Input(shape=block4_shape)
    emo_block4_model = emo_block4(emo_input=block4_input)
    value7 = emo_block4_model.get_weights()
    emo_block4_model.load_weights("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/block_weights/imagenet_block4.h5")
    value8 = emo_block4_model.get_weights()
    emo_output = emo_block4_model(emo_output)

    block5_shape = (14, 14, 1024,)
    block5_input = Input(shape=block5_shape)
    emo_block5_model = emo_block5(emo_input=block5_input)
    value9 = emo_block5_model.get_weights()
    emo_block5_model.load_weights("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/block_weights/imagenet_block5.h5")
    value10 = emo_block5_model.get_weights()
    emo_output = emo_block5_model(emo_output)

    emo_output = AveragePooling2D((7, 7), name='avg_pool_emo')(emo_output)
    emo_output = Flatten(name='flatten_emo')(emo_output)
    emo_output = Dense(8, activation='softmax', kernel_regularizer=regularizers.l2(0.01), name='emo_output')(emo_output)
    my_model = Model(inputs=emo_input, outputs=emo_output)
    # my_model.load_weights("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/finetune_weights/emo/emo8_23255_lr_-4_finetune_fully-connect+block21(-5).h5", by_name = True)
################### test ########################
    # f1 = h5py.File("/home/yujun/PycharmProjects/Emotion_Recognition/emo/emo8_23255_split.h5", "r")
    # # emo_X_train = f1["images_train"][:]
    # # emo_Y_train = f1["labels_train"][:]
    # emo_X_test = f1["images_test"][:]
    # emo_Y_test = f1["labels_test"][:]
    # f1.close()
    # emo_score = my_model.predict(emo_X_test[:, 0:224, 0:224, :], verbose=1)
    # print accuracy(emo_score.tolist(), emo_Y_test.tolist())
    # del emo_X_test
    # del emo_Y_test
    gc.collect()
    # emo_block1_model.save_weights("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/block_weights/emo_23255_finetune_block1.h5")
    # emo_block2_model.save_weights("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/block_weights/emo_23255_finetune_block2.h5")
    # emo_block3_model.save_weights("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/block_weights/emo_23255_finetune_block3.h5")
    # emo_block4_model.save_weights("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/block_weights/emo_23255_finetune_block4.h5")
    # emo_block5_model.save_weights("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/block_weights/emo_23255_finetune_block5.h5")

    # f1 = h5py.File("/home/yujun/PycharmProjects/Emotion_Recognition/emo/emo8_23255_split.h5", "r")
    # emo_X_train = f1["images_train"][:]
    # emo_Y_train = f1["labels_train"][:]
    # f1.close()
    # emo_X_test = f1["images_test"][:]
    # emo_Y_test = f1["labels_test"][:]
    # f1.close()
    # emo_X_real_train, emo_X_validation, emo_Y_real_train, emo_Y_validation = data_split(emo_X_train, emo_Y_train, 0.2)
    # del emo_X_train
    # del emo_Y_train
    # gc.collect()

    # f1 = h5py.File("/home/yujun/PycharmProjects/Emotion_Recognition/emo/emo_23255_training_split.h5", "r")
    # # f1["images_train"] = emo_X_real_train
    # # f1["labels_train"] = emo_Y_real_train
    # # f1["images_val"] = emo_X_validation
    # # f1["labels_val"] = emo_Y_validation
    # emo_X_real_train = f1["images_train"][:]
    # emo_Y_real_train = f1["labels_train"][:]
    # emo_X_validation = f1["images_val"][:]
    # emo_Y_validation = f1["labels_val"][:]
    # f1.close()
    f1 = h5py.File("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_WORK/datasets/emo8_balance.h5", "r")
    emo_X_real_train = f1["images_train"][:]
    emo_Y_real_train = f1["labels_train"][:]
    emo_X_validation = f1["images_val"][:]
    emo_Y_validation = f1["labels_val"][:]
    emo_X_test = f1["images_test"][:]
    emo_Y_test = f1["labels_test"][:]
    f1.close()
    del emo_X_test
    del emo_Y_test
    gc.collect()
    # for layer in my_model.layers[:3]:
    #     layer.trainable = False
    #
    sgd = SGD(lr=4e-4, decay=1e-13, momentum=0.92, nesterov=True)
    my_model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy']) # kullback_leibler_divergence
    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=8, verbose=1, mode='min',
                                     epsilon=1e-4, cooldown=0, min_lr=0)
    Checkpoint = ModelCheckpoint(filepath='/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/single_weights/emo8_22186_lr_4-4_no_finetune.h5',
                                     monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
    earlyStop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')
    Emo_history = my_model.fit_generator(generator=for_generate(emo_X_real_train, emo_Y_real_train, batch_size=16),
                                        validation_data=for_generate(emo_X_validation, emo_Y_validation, batch_size=16),
                                        shuffle = True, steps_per_epoch=888, validation_steps=444, epochs=500, verbose=2,
                                        callbacks=[ReduceLR, Checkpoint, earlyStop])    #1022,256;emo:989 248
    #
    pass