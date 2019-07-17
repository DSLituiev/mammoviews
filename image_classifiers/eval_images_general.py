
# coding: utf-8
import os
import yaml
import sys
import numpy as np
import pandas as pd
from PIL import Image

from functools import partial
from keras.optimizers import Adam
from keras.models import load_model
import image
#from image import ImageDataGenerator
#from inception_short import get_model

##############################################3
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_HOME'] = '/usr/local/cuda-8.0'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["PYTHONHASHSEED"]='0'

def batch_iterator(paths, batch_size=8, target_side=99, target_size=None,
                   preprocessing_function=None, 
                   img_loader=partial(image.load_img_opencv, color_mode='rgb')):
    batchx = []
    batchmeta = []
    ii = 0
    if (target_size is None) and (target_side is not None):
        target_size = (target_side, target_side)
    for filepath in (paths):
        try:
            img = img_loader(filepath, target_size=target_size,)
            #import ipdb; ipdb.set_trace()
            #img = Image.open(filepath).convert('F')
            #if target_size is not None:
            #    img = img.resize(target_size)
            #img = np.asarray(img)/2**8
            batchx.append(img)
            batchmeta.append(os.path.basename(filepath))
        except Exception as ex:
            raise ex

        ii+=1
        if ii % batch_size == 0 or ii == len(paths):
            #batchx = np.stack([np.stack(batchx)]*3,axis=-1)
            batchx = np.stack(batchx, axis=0)
            if preprocessing_function is not None:
                batchx = preprocessing_function(batchx)
            yield batchmeta, batchx
            batchx = []
            batchmeta = []

##############################################
## AUGMENT BY FLIPPING L-R?
fliplr = True

if fliplr:
    preprocessing_function = lambda x: x[...,::-1,:]
    flipsuffix = 'fliplr'
else:
    preprocessing_function = None
    flipsuffix = 'orig'
##############################################
## CONSTRUCT A LIST OF PNG FILE PATHS
# this script can be modified to read DICOMs: 
# replace `read_img_opencv` with your favorite reader in `batch_generator()`

fnmeta = "test.csv.gz"
# df = pd.read_csv(fnmeta)
# pngdir = "/media/exx/tron/2017-07-png-jae/"
# df["png"] = df["id"].map(lambda x: os.path.join(pngdir, x+".png"))
# png_list = df["png"].values
png_list = ['data/test.dcm']

## FORMAT AN OUTPUT FILE 
fnbase = os.path.basename(fnmeta).replace(".gz","").replace(".csv","")
fnoutpred = os.path.join(os.path.dirname(fnmeta),
                         '{}-spotmag_img_prediction-{}-{}.csv'.format(
                             fnbase, 'general',
                             flipsuffix))

##############################################
## LOAD WEIGHTS AND OTHER INFERENCE SETTINGS
batch_size = 128 
WEIGHTFILE = "e5ce2d69b035975cb5336cec0da9a32a/model-272-general-e5ce2d69b035975cb5336cec0da9a32a.hdf5"
indir = os.path.dirname(WEIGHTFILE)

print(WEIGHTFILE)

with open(os.path.join(indir, "checkpoint.info")) as chkpt_fh:
    prms = yaml.load(chkpt_fh)
    print("\n".join(["%s\t%s" %(kk,vv) for kk,vv in prms.items()]),)

print("loading weights from:\t%s" % WEIGHTFILE)
model = load_model(WEIGHTFILE)
##############################################

#model = get_model(n_classes=prms["n_classes"],
#              final_activation=prms["final_activation"],
#              ndense=prms["ndense"],
#              base_trainable=prms["base_trainable"])


prms["loss"] = '{}_crossentropy'.format( prms["class_mode"] )
model.compile(optimizer=Adam(lr=prms["lr"]), loss=prms["loss"], metrics=['accuracy'])
    

print("SAVING TO", fnoutpred)
try:
    os.unlink(fnoutpred)
except:
    print("%s\tnot found" % fnoutpred)
    pass

#################################
## set image loader
# for PNGs:
#img_loader=partial(image.load_img_opencv, color_mode='rgb')

# for DICOMs:
img_loader=image.load_pydicom

biter = batch_iterator(png_list, batch_size=batch_size,
                       img_loader=img_loader,
                       target_size = (99,99),
                       preprocessing_function=preprocessing_function)

#import ipdb
#ipdb.set_trace()

kwargs = dict(header=True)

for nn, (filenames_, batch) in enumerate(biter):
    yscore = model.predict(batch)
    index = [ff.split("/")[-1].replace(".png","") for ff in filenames_]
    dfout = pd.Series(yscore.ravel(), 
            index=index)
    dfout.to_csv(fnoutpred, **kwargs)
    kwargs = dict(mode='a', header=None)
    print(nn)
    print(dfout)

print("DONE")
