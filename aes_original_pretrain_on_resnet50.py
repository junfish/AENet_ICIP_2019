from keras import Input
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras import initializers
from keras.applications.resnet50 import ResNet50
from keras.layers.convolutional import ZeroPadding2D, Conv2D, MaxPooling2D
from keras import layers
from keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import concatenate
from keras import regularizers
from keras.models import Model
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from resnet50_block import block1, block2, block3, block4, block5
import numpy as np
import random
import h5py
import os
import gc
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU()(x)
    x = ELU()(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU()(x)
    x = ELU()(x)


    x = Conv2D(filters3, (1, 1), kernel_regularizer=regularizers.l2(0.01), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU()(x)
    x = ELU()(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters

    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, kernel_regularizer=regularizers.l2(0.01),
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU()(x)
    x = ELU()(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.01),
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU()(x)
    x = ELU()(x)

    x = Conv2D(filters3, (1, 1), kernel_regularizer=regularizers.l2(0.01), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_regularizer=regularizers.l2(0.01),
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU()(x)
    x = ELU()(x)
    return x



def aes_Resnet50(input_tensor):

    bn_axis = 3


    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = PReLU()(x)
    x = ELU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    # x = Dropout(0.05)(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    # x = Dropout(0.05)(x)

    x = Flatten()(x)
    x = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01),name='fc2')(x)


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    model = Model(input_tensor, x, name='aes_resnet')
    return model

def data_split(X, Y, test_rate):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_rate)
    return X_train, X_test, Y_train, Y_test

def for_generate(X, Y, batch_size):
    while 1:
        # f_x = open(path1)
        # f_y = open(path2)
        cnt = 0
        x_batch =[]
        y_batch =[]
        for num in Y:
            # create Numpy arrays of input data
            # and labels, from each line in the file
            # x, y = process_line(line)
            index = random.randint(0, len(X) - 1)
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

def accuracy(predict_list, truth_list):
    all = len(truth_list)
    acc = 0
    for (predict_list_element, truth_list_element) in zip(predict_list, truth_list):
        if predict_list_element.index(max(predict_list_element)) == truth_list_element.index(max(truth_list_element)):
            acc += 1
    return (float(acc) / float(all))

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    input_shape = (224, 224, 3,)
    input = Input(shape=input_shape)
    aes_input = Input(shape=input_shape, name='aes_input')
    # model = aes_Resnet50(input_tensor = aes_input)  # weights = "imagenet"
    # model.load_weights("/home/yujun/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5", by_name = True)
    aes_block1_model = block1(input)
    aes_output = aes_block1_model(aes_input)

    block2_shape = (55, 55, 64,)
    block2_input = Input(shape=block2_shape)
    aes_block2_model = block2(block2_input)
    aes_output = aes_block2_model(aes_output)

    block3_shape = (55, 55, 256,)
    block3_input = Input(shape=block3_shape)
    aes_block3_model = block3(block3_input)
    aes_output = aes_block3_model(aes_output)

    block4_shape = (28, 28, 512,)
    block4_input = Input(shape=block4_shape)
    aes_block4_model = block4(block4_input)
    aes_output = aes_block4_model(aes_output)

    block5_shape = (14, 14, 1024,)
    block5_input = Input(shape=block5_shape)
    aes_block5_model = block5(block5_input)
    aes_output = aes_block5_model(aes_output)

    aes_output = AveragePooling2D((7, 7), name='avg_pool_aes')(aes_output)
    aes_output = Flatten(name='flatten_aes')(aes_output)
    aes_output = Dense(8, activation='softmax', kernel_regularizer=regularizers.l2(0.01), name='emo_output')(aes_output)
    my_model = Model(inputs=aes_input, outputs=aes_output)
    # model.load_weights('/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/nor_aes-5.h5')
    # model.trainable = True
    # model.layers[-1].kernel_initializer = initializers.glorot_normal()
    ###################### load data ##########################
    # f1 = h5py.File("/home/yujun/PycharmProjects/Emotion_Recognition/emo/emo_256.h5", "r")
    # emo_X_train = f1["train_images"][:]
    # emo_Y_train = f1["train_labels"][:]
    # emo_X_test = f1["test_images"][:]
    # emo_Y_test = f1["test_labels"][:]
    # emo_X_real_train, emo_X_validation, emo_Y_real_train, emo_Y_validation = data_split(emo_X_train, emo_Y_train, 0.2)
    # f2 = h5py.File("/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/aes_pretrain_2class_float32.h5")
    f2 = h5py.File("/home/yujun/PycharmProjects/Emotion_Recognition/Aes_mysql/multi_label_22086.h5", "r")
    # f3 = h5py.File("/home/yujun/PycharmProjects/Emotion_Recognition/ava/test_data.hdf5")
    aes_images = f2["images"][:]
    aes_labels = f2["aes_labels"][:]
    f2.close()
    # f1 = h5py.File("/home/yujun/PycharmProjects/NEW_WORK/datasets/aes_train_data.h5")
    # aes_train_images = f1["images_train"][:]
    # aes_train_labels = f1["labels_train"][:]
    # f1.close()
    aes_mean = np.mean(aes_images, axis=0)
    aes_std = np.std(aes_images, axis = 0, ddof = 1)
    aes_images = (aes_images - aes_mean) / aes_std
    # aes_X_1, aes_X_5, aes_Y_1, aes_Y_5 = data_split(aes_images, aes_labels, 0.2)
    # aes_X_1, aes_X_4, aes_Y_1, aes_Y_4 = data_split(aes_X_1, aes_Y_1, 0.25)
    # aes_X_1, aes_X_3, aes_Y_1, aes_Y_3 = data_split(aes_X_1, aes_Y_1, 0.3333333333333333)
    # aes_X_1, aes_X_2, aes_Y_1, aes_Y_2 = data_split(aes_X_1, aes_Y_1, 0.5)
    # f1 = h5py.File("/home/yujun/PycharmProjects/Emotion_Recognition/Cross-stitch/data/aes_emo_22086_5-fold.h5", "w")
    # f1["images_fold_1"] = aes_X_1
    # f1["images_fold_2"] = aes_X_2
    # f1["images_fold_3"] = aes_X_3
    # f1["images_fold_4"] = aes_X_4
    # f1["images_fold_5"] = aes_X_5
    # f1["labels_fold_1"] = aes_Y_1
    # f1["labels_fold_2"] = aes_Y_2
    # f1["labels_fold_3"] = aes_Y_3
    # f1["labels_fold_4"] = aes_Y_4
    # f1["labels_fold_5"] = aes_Y_5
    # f1.close()
    # aes_train_labels = f1["labels_train"][:]
    # f1.close()
    aes_X_real_train, aes_X_validation, aes_Y_real_train, aes_Y_validation = data_split(aes_images, aes_labels, 0.2)
    # sss = np.concatenate((aes_X_real_train, aes_X_validation), axis = 0)
    del aes_images
    del aes_labels
    gc.collect()
    ####################### fine-tuned to train emotion model ###############
    # model.load_weights('/home/yujun/PycharmProjects/Emotion_Recognition/weights/aes_train_zero_centre.h5')
    # model.trainable = True
    # sss = emo_model.get_layer(name = "conv1").get_weights()
    # test_batches = len(emo_Y_test) / 32 + 1
    # score = model.predict(emo_X_test[:, 0:224, 0:224, :], verbose=1)
    # print accuracy(score.tolist(), emo_Y_test.tolist())
    sgd = SGD(lr=2e-5, decay=1e-10, momentum=0.94, nesterov=True)
    # adam = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
    model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=["accuracy"])
    # history = model.fit_generator(generator=for_generate(aes_X_real_train, aes_Y_real_train, batch_size=16),
    #                               validation_data=for_generate(aes_X_validation, aes_Y_validation, batch_size=16),
    #                               steps_per_epoch=5110, validation_steps=1278, epochs=100, verbose=2)
    # model.save_weights('/home/yujun/PycharmProjects/NEW_WORK/weights/aes_pretrain_adam.h5')
    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8,
                                 verbose=1, mode='min', epsilon=1e-5, cooldown=1, min_lr=0)

    Checkpoint = ModelCheckpoint(filepath='/home/yujun/PycharmProjects/Emotion_Recognition/NEW_NEW_WORK/weights/nor_aes-5.h5',
                                 monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)

    earlyStop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')

    history = model.fit_generator(generator=for_generate(aes_X_real_train, aes_Y_real_train, batch_size=64),
                                  validation_data=for_generate(aes_X_validation, aes_Y_validation, batch_size=64),
                                  steps_per_epoch=1022, validation_steps=256, epochs=500, verbose=2,
                                  callbacks=[ReduceLR, Checkpoint, earlyStop])
    pass