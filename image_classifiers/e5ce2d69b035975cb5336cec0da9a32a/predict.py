
# coding: utf-8
import sys
import pandas as pd
sys.path.append('../..')

from inception_short import get_model, get_num_files, get_class_weights
from keras.optimizers import Adam
from image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from checkpoint_utils import CSVWallClockLogger, lr_cyclic_schedule
from shutil import copy2
from functools import partial

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

import os
import yaml
import numpy as np
import keras
from hashlib import md5
os.environ["PYTHONHASHSEED"]='0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_HOME'] = '/usr/local/cuda-8.0'

if os.environ["CUDA_VISIBLE_DEVICES"] == '':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'


indir = "./"

import yaml
with open(os.path.join(indir, "checkpoint.info")) as chkpt_fh:
    prms = AttrDict(yaml.load(chkpt_fh))
    print("\n".join(["%s\t%s" %(kk,vv) for kk,vv in prms.items()]),)

weightfile = os.environ["WFILE"]
#weightfile = "model.175-0.068012.hdf5"
prms['weightfile'] =  weightfile
prms['weightfile'] = os.path.join(indir, prms['weightfile'])
prms['weightfile']


# In[6]:


prms["loss"] = '{}_crossentropy'.format( prms.class_mode )
print("loss:", prms["loss"])

# CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model.{epoch:02d}-{val_loss:2f}.hdf5')

SAMPLES_PER_EPOCH = get_num_files(prms.data_train)
STEPS_PER_EPOCH = SAMPLES_PER_EPOCH // prms.batch_size

print('='*50)
print("samples per epoch in the train set: %d" % SAMPLES_PER_EPOCH)
print("steps per epoch in the train set: %d" % STEPS_PER_EPOCH)
print('='*50)
#########################################

if prms.weightfile:
    print("LOADING WEIGHTS FROM:\t%s" % prms.weightfile)
#     model.load_weights(prms.weightfile)
    model = load_model(prms.weightfile)


# In[22]:


flowfromdir_params = dict(
#     color_mode = "grayscale",
    target_size=prms.target_size,
    batch_size=prms.batch_size,
    class_mode=prms.class_mode,
    classes=prms.classes,
    seed=prms.seed)

norm_params = dict(
        #rescale=prms.scaleup,
        samplewise_center=prms.samplewise_center,
        samplewise_std_normalization=prms.samplewise_center,
        featurewise_center=False,
        featurewise_std_normalization=False,
        zca_whitening=False,
        )


# In[23]:


train_datagen = ImageDataGenerator(**norm_params)

train_datagen.preprocessing_function = lambda x: x[...,::-1,:]#*2**-8
datagen_train_output = train_datagen.flow_from_directory(
    prms.data_train,
    #stratify = prms.oversampling,
    #sampling_factor=prms.sampling_factor,
    #oversampling=prms.oversampling,
    shuffle=False, **flowfromdir_params)
SAMPLES_PER_EPOCH = len(datagen_train_output.filenames)
STEPS_PER_EPOCH = int(np.ceil(SAMPLES_PER_EPOCH / prms.batch_size))

##########################################
def get_predictions(data_dir, 
                    preprocessing_function = lambda x:x,
                    model=model):
    if isinstance(preprocessing_function, str):
        if preprocessing_function == 'fliplr':
            preprocessing_function = lambda x: x[...,::-1,:]
        elif preprocessing_function in ('identity', 'orig'):
            preprocessing_function = lambda x:x
        else:
            raise ValueError('unknown preprocessing_function:\t%s' 
                             % preprocessing_function)

    val_datagen = ImageDataGenerator(**norm_params)
    val_datagen.preprocessing_function = preprocessing_function
    datagen_val_output = val_datagen.flow_from_directory(
            data_dir,
            shuffle=False, **flowfromdir_params)

    gen_ = datagen_val_output 
    yhat = model.predict_generator(gen_,
                          steps=len(gen_),
                          verbose=1,)

    dfdict = {"scores_%d"%nn : yy for nn, yy in enumerate(yhat.T)}
    dfdict.update({ "files":gen_.filenames, "label": gen_.classes})
    dfres = pd.DataFrame(dfdict)
    return dfres
##########################################
#                HOLDOUT 
##########################################
data_holdout = '/data/UCSF_MAMMO/2018-02-png/withx_valset_4000_val'
dfres = get_predictions(
                data_holdout, 
                preprocessing_function = lambda x:x,
                model=model)
dfres.to_csv("predictions_val.csv", index=False)
##########################################
preprocessing_function = lambda x: x[...,::-1,:]
dfres = get_predictions(
                data_holdout, 
                preprocessing_function = preprocessing_function,
                model=model)

dfres.to_csv("predictions_val_fliplr.csv", index=False)
##########################################
#                Test 
##########################################

dfres = get_predictions(
                prms.data_val,
                preprocessing_function = lambda x:x,
                model=model)
dfres.to_csv("predictions_test.csv", index=False)
##########################################

preprocessing_function = lambda x: x[...,::-1,:]
dfres = get_predictions(
                prms.data_val,
                preprocessing_function = preprocessing_function,
                model=model)
dfres.to_csv("predictions_test_fliplr.csv", index=False)
##########################################
#                 TRAIN
##########################################
dfres = get_predictions(
                prms.data_train,
                preprocessing_function = lambda x:x,
                model=model)
dfres.to_csv("predictions_train.csv", index=False)
##########################################
preprocessing_function = lambda x: x[...,::-1,:]
dfres = get_predictions(
                prms.data_train,
                preprocessing_function = preprocessing_function,
                model=model)
dfres.to_csv("predictions_train_fliplr.csv", index=False)

