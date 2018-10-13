#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:00:55 2017

@author: dlituiev
"""

import os
from collections import Counter
from functools import partial
from itertools import product

import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GaussianNoise, Input
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization, Input
from keras.optimizers import Adam

#########################################
def get_num_files(parentdir):
    numfiles = 0
    for dd in os.scandir(parentdir):
        dd = os.path.join(parentdir, dd)
        if os.path.isdir(dd):
            numfiles+= sum((1 for ff in os.scandir(dd)))
    return numfiles
#########################################
#########################################
#          SET UP THE NETWORK
#########################################
def get_model(n_classes, final_activation,
              ndense=512, dropout=0.5,
              weights='imagenet',
              input_shape = [None, None, 3],
              gaussian_noise_sigma = None,
              input_tensor = None,
              base_trainable=False):

    if input_shape:
        input_tensor = Input(shape = input_shape)
    if gaussian_noise_sigma is not None:
        input_tensor = GaussianNoise(gaussian_noise_sigma)(input_tensor)
    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, include_top=False,
                             input_tensor = input_tensor,
                            )
    # get third Concatenation layer and crop the network on it:
    cc=0
    poptherest = False
    for nn, la in enumerate(base_model.layers):
        if type(la) is keras.layers.Concatenate:
            if cc==3:
                x = la.output
                break
            cc+=1
    base_model.layers = base_model.layers[:nn+1]

    #x = [la.output for la in base_model.layers if type(la) is keras.layers.Concatenate][3]
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dropout(dropout)(x)

    if ndense>0:
        x = Dense(ndense, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(n_classes, activation=final_activation)(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    if not base_trainable:
        for layer in base_model.layers:
            layer.trainable = False

    last_module_index = [nn for nn,la  in enumerate(model.layers) if type(la) is keras.layers.Concatenate][-2]

    for layer in model.layers[last_module_index:]:
        layer.trainable = True
    return model


def get_class_weights(datagen_val_output):
    counter = Counter(datagen_val_output.classes)
    print("distribution of labels in {}:\n{}".format(datagen_val_output.directory, str(counter)))
    for kk,vv in counter.items():
        counter[kk] = vv+1

    max_val = float(max(counter.values()))

    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}                     
    return class_weights



def w_categorical_crossentropy(weights):
    def _w_categorical_crossentropy(y_true, y_pred, weights):
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.expand_dims(y_pred_max, 1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):

            final_mask += (K.cast(weights[c_t, c_p],K.floatx()) *
                           K.cast(y_pred_max_mat[:, c_p] ,K.floatx()) *
                           K.cast(y_true[:, c_t],K.floatx())
                          )
        return K.categorical_crossentropy(y_pred, y_true) * final_mask

    ncce = partial(_w_categorical_crossentropy, weights=weights)
    ncce.__name__ ='w_categorical_crossentropy'
    return ncce


if __name__ == '__main__':
    import numpy as np
    import keras
    #csv_path = CHECKPOINTS_BASE + ".log.csv"
    #csv_callback = keras.callbacks.CSVLogger(csv_path, separator=',', append=False)
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['CUDA_HOME'] = '/usr/local/cuda-8.0'
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    NDENSE=256 #512
    BATCH_SIZE = 128
    NB_EPOCH = 20
    DATA_AUGMENTATION = True
    SEED=0
    CLASS_MODE = 'binary' # 'categorical'
    LOSS = '{}_crossentropy'.format(CLASS_MODE)
    N_CLASSES = 1
    FINAL_ACTIVATION = 'sigmoid'
    LR = 0.0001
    SAMPLEWISE_CENTER = False #True

    TARGET_SIDE = 99
    TARGET_SIZE = [TARGET_SIDE]*2

    BASE_TRAINABLE=False
    CHECKPOINT_DIR = "./modelstate_withx_negloglr{:d}_ndense{:d}_imsize{:d}{}/" .format(
                    int(-np.log10(LR)),
                    NDENSE,
                    TARGET_SIDE,
                    "" if not BASE_TRAINABLE else "_base_trainable"
                    )
    CHECKPOINT_PATH = CHECKPOINT_DIR + 'model.{epoch:02d}-{val_loss:2f}.hdf5'

    WEIGHTFILE = None # "./modelstate_withx_negloglr4_ndense256/model.39-0.060567.hdf5" # None # "./modelstate_withx/model.03-0.067136.hdf5"
    # "modelstate_laplace_inv_weights_2/model.10-0.014968.hdf5" #CHECKPOINT_DIR + "model.10-0.019602.hdf5"
    INIT_EPOCH=0
    # indir = "/data/dlituiev/learn_spotmag_from_images/modelstate/"
    # find_min_loss_checkpoint(indir)


    DATA_TRAIN = '/data/UCSF_MAMMO/2017-07-png/withx_valset_4000_train/'
    DATA_VAL = '/data/UCSF_MAMMO/2017-07-png/withx_valset_4000_test/'
    SAMPLES_PER_EPOCH = get_num_files(DATA_TRAIN)
    STEPS_PER_EPOCH = SAMPLES_PER_EPOCH // BATCH_SIZE

    CLASSES = ["normal", "special"]

    VALIDATION_STEPS = get_num_files(DATA_VAL) // BATCH_SIZE
    print('='*50)
    print("validation steps", VALIDATION_STEPS)
    print("samples per epoch in the train set: %d" % SAMPLES_PER_EPOCH)
    print("steps per epoch in the train set: %d" % STEPS_PER_EPOCH)
    print('='*50)
    #########################################
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_loss', verbose=1,
            save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callbacks_list =[checkpoint]

    #########################################
    model = get_model(n_classes=N_CLASSES,
                      final_activation=FINAL_ACTIVATION,
                      ndense=NDENSE,
                      dropout=0.5,
                      base_trainable=BASE_TRAINABLE)


    #from keras.utils import plot_model
    #plot_model(model, to_file='model.png')


    model.compile(optimizer=Adam(lr=LR), loss=LOSS, metrics=['accuracy'],
                  callbacks = [csv_callback])
    #########################################
    if WEIGHTFILE:
        print("loading weights from:\t%s" % WEIGHTFILE)
        model.load_weights(WEIGHTFILE)

    print('Using real-time data augmentation.')

    flowfromdir_params = dict(
        #color_mode = "grayscale",
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        classes=CLASSES,
        seed=SEED)

    train_datagen = ImageDataGenerator(
        samplewise_center=SAMPLEWISE_CENTER,
        samplewise_std_normalization=SAMPLEWISE_CENTER,
        featurewise_center=False,
        featurewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False)

    val_datagen = ImageDataGenerator()

    datagen_train_output = train_datagen.flow_from_directory(
        DATA_TRAIN, shuffle=True, **flowfromdir_params)

    datagen_val_output = val_datagen.flow_from_directory(
        DATA_VAL, shuffle=False, **flowfromdir_params)

    class_weights = get_class_weights(datagen_val_output)

    model.fit_generator(datagen_train_output,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          epochs=NB_EPOCH, verbose=1,
                          validation_data=datagen_val_output,
                          validation_steps=VALIDATION_STEPS,
                          #class_weight='auto',
                          class_weight=class_weights,
                          callbacks=callbacks_list,
                          initial_epoch=INIT_EPOCH)



    #model.predict()
