from inception_short import get_model, get_num_files, get_class_weights
from keras.optimizers import Adam
from image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from checkpoint_utils import CSVWallClockLogger
from shutil import copy2

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

import sys
import os
import yaml
import numpy as np
import keras
from hashlib import md5
os.environ["PYTHONHASHSEED"]='0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_HOME'] = '/usr/local/cuda-8.0'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

prms = AttrDict(
    dropout=0.5,
    base_trainable=False,
    horizontal_flip = True,
    vertical_flip = False,
    zoom_range = [0.8, 1.2],
    rotation_range = 30,
    fill_mode='reflect',
    ndense=0,
    batch_size = 16,
    init_epoch=0,
    nb_epoch = 500,
    data_augmentation = True,
    rescale = 1, #2**-8,
    #contrast = 0.9,
    truncate_quantile = None,#0.001,
    ztransform = True,
    oversampling = False,
    #sampling_factor = [1, 4],
    seed=1,
    width_shift_range = 0.125,
    height_shift_range = 0.125,
    class_mode =  'categorical', # 'binary', #
    n_classes = 2,
    final_activation = "softmax", # 'sigmoid',
    lr = 1e-3,
    samplewise_center = False, #True
    target_side = 299,
    #weights = None,
    weightfile = None, #"checkpoints/6a1a17e4bcaabe458c145fd64dec0322/model.31-1.290145.hdf5",
    #"checkpoints/6a1a17e4bcaabe458c145fd64dec0322/model.59-1.676424.hdf5",
    data_train = '/data/UCSF_MAMMO/2018-02-png/each_class_4189_train/',
    data_val = '/data/UCSF_MAMMO/2018-02-png/each_class_4189_test/',
    classes = ["normal", "wire"],
    class_weights=[1, 1],
    ReduceLROnPlateau = dict(
        monitor='val_loss',
        factor=1/2,
        patience=32*2,
        verbose=0,
        mode='auto', epsilon=0.001,
        cooldown=8,
        min_lr=1e-12,
        ),
)


paramhash = md5(str(prms).encode()).hexdigest()

prms["target_size"] = [ prms.target_side ]*2

CHECKPOINT_DIR = "checkpoints/" + paramhash + "/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print("SAVING TO:\t%s" % CHECKPOINT_DIR)
# copy the script to the checkpoint directory
copy2(os.path.abspath(__file__), CHECKPOINT_DIR)
with open(os.path.join(CHECKPOINT_DIR, "checkpoint.info"), "w+") as outfh:
    yaml.dump(dict(prms), outfh, default_flow_style=False)
prms["loss"] = '{}_crossentropy'.format( prms.class_mode )

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model.{epoch:02d}-{val_loss:2f}.hdf5')

SAMPLES_PER_EPOCH = get_num_files(prms.data_train)
STEPS_PER_EPOCH = SAMPLES_PER_EPOCH // prms.batch_size

print('='*50)
print("samples per epoch in the train set: %d" % SAMPLES_PER_EPOCH)
print("steps per epoch in the train set: %d" % STEPS_PER_EPOCH)
print('='*50)
#########################################
checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='auto', period=1)

csv_path = os.path.join(CHECKPOINT_DIR, "progresslog.csv")
csv_callback = CSVWallClockLogger(csv_path, separator=',', append=False)


callback_list = [checkpoint, csv_callback]


if ("ReduceLROnPlateau" in prms) and prms["ReduceLROnPlateau"]:
            callback_list.append(ReduceLROnPlateau(**prms["ReduceLROnPlateau"]))

#########################################
model = get_model(n_classes=prms.n_classes,
                  final_activation=prms.final_activation,
                  ndense=prms.ndense,
                  #weights = prms.weights,
                  dropout=prms.dropout,
                  base_trainable=prms.base_trainable)


#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

if __name__ == '__main__':
    model.compile(optimizer=Adam(lr=prms.lr), loss=prms.loss,
                  metrics=['accuracy', #acc_0, acc_1,# acc_2, acc_3, acc_4
                      ],
                  )
    #########################################
    if prms.weightfile:
        print("loading weights from:\t%s" % prms.weightfile)
        model.load_weights(prms.weightfile)
    
    #########################################
    print('Using real-time data augmentation.')

    flowfromdir_params = dict(
        #color_mode = "grayscale",
        target_size=prms.target_size,
        batch_size=prms.batch_size,
        class_mode=prms.class_mode,
        classes=prms.classes,
        seed=prms.seed)
    norm_params = dict(
            rescale=prms.rescale,
            samplewise_center=prms.samplewise_center,
            samplewise_std_normalization=prms.samplewise_center,
            featurewise_center=False,
            featurewise_std_normalization=False,
            zca_whitening=False,
            z_transform = prms.ztransform,
            )

    def _ztransform(x):
        return (x-np.mean(x)) / np.std(x)

    if 'preprocessing_function' in prms:
        if prms.preprocessing_function=='ztransform':
            preprocessing_function = _ztransform
        elif prms.preprocessing_function=='m1p1':
            preprocessing_function = lambda x: x/128.0 - 1
        else:
            raise ValueError("unknown preprocessing_function")
    else:
        preprocessing_function = lambda x: x


    if prms.data_augmentation:

        print('Using real-time data augmentation.')
        train_datagen = ImageDataGenerator(
            zoom_range=prms.zoom_range,
            fill_mode=prms.fill_mode,
            rotation_range = prms.rotation_range,
            width_shift_range = prms.width_shift_range,
            height_shift_range = prms.height_shift_range,
            horizontal_flip=prms.horizontal_flip,
            vertical_flip=prms.vertical_flip,
            contrast = prms.contrast if "contrast" in prms else None,
            truncate_quantile = prms.truncate_quantile,
            #histeq_alpha=prms.histeq_alpha,
            **norm_params)
    else:
        train_datagen = ImageDataGenerator(**norm_params)

    val_datagen = ImageDataGenerator(**norm_params)

    datagen_train_output = train_datagen.flow_from_directory(
        prms.data_train, 
        stratify = prms.oversampling,
        sampling_factor=prms.sampling_factor if prms.oversampling else None,
        oversampling=prms.oversampling,
        shuffle=True, **flowfromdir_params)

    datagen_val_output = val_datagen.flow_from_directory(
        prms.data_val, shuffle=False, **flowfromdir_params)

    #VALIDATION_STEPS = get_num_files(prms.data_val) // prms.batch_size
    VALIDATION_STEPS = np.ceil(len(datagen_val_output.filenames)/prms['batch_size'])
    print("validation steps", VALIDATION_STEPS)
    #########################################
    if prms.class_weights == 'auto':
        class_weights = get_class_weights(datagen_val_output)
    else:
        class_weights = prms.class_weights

    model.fit_generator(datagen_train_output,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          epochs=prms.nb_epoch, verbose=1,
                          validation_data=datagen_val_output,
                          validation_steps=VALIDATION_STEPS,
                          #class_weight='auto',
                          class_weight=class_weights,
                          callbacks=callback_list,
                          initial_epoch=prms.init_epoch)

    datagen_val_output = val_datagen.flow_from_directory(
        prms.data_val, shuffle=False, **flowfromdir_params)

    print("""loss\t%.4f
    accuracy\t%.4f\n""" %
      tuple(model.evaluate_generator(datagen_val_output,
                                     steps=VALIDATION_STEPS,
                                     workers=1,
                                    pickle_safe=True)))


    #model.predict()
