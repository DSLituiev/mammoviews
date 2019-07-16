
# coding: utf-8
import os
import yaml
import sys
import numpy as np
import pandas as pd
from PIL import Image
from checkpoint_utils import CheckpointParser
sys.path.append("../learn_spotmag_from_dicom_headers/")

from keras.optimizers import Adam
from keras.models import load_model
from image import load_img_opencv
#from image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
#from inception_short import get_model

##############################################3
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_HOME'] = '/usr/local/cuda-8.0'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["PYTHONHASHSEED"]='0'

def batch_iterator(paths, batch_size=8, target_side=99, target_size=None,
                   preprocessing_function=None):
    batchx = []
    batchmeta = []
    ii = 0
    if (target_size is None) and (target_side is not None):
        target_size = (target_side, target_side)
    for pp in (paths):
        try:
            img = load_img_opencv(pp, color_mode='rgb', 
                                  target_size=target_size,)
            #import ipdb; ipdb.set_trace()
            #img = Image.open(pp).convert('F')
            #if target_size is not None:
            #    img = img.resize(target_size)
            #img = np.asarray(img)/2**8
            batchx.append(img)
            batchmeta.append(os.path.basename(pp))
        except Exception as ex:
            raise ex
        ii+=1
        if ii%batch_size == 0:
            #batchx = np.stack([np.stack(batchx)]*3,axis=-1)
            batchx = np.stack(batchx, axis=0)
            if preprocessing_function is not None:
                batchx = preprocessing_function(batchx)
            yield batchmeta, batchx
            batchx = []
            batchmeta = []

##############################################3
batch_size = 128 
#indir = "checkpoints/94bec262cbc23509c9e27eb3d427c5a2/"
#"checkpoints/d1baf6b2a9558ebeb197140db121767b/model.44-0.145893.hdf5"
#indir = "checkpoints/1cfdc49901cec8c56b66fb200de72f57"
indir = "checkpoints/e5ce2d69b035975cb5336cec0da9a32a"

dfchkpt = pd.DataFrame([[os.path.join(indir,x.name)] + CheckpointParser(x.name)() + [x.stat().st_atime] for x in os.scandir(indir) if x.name.endswith(".hdf5")],
                     columns=["filename", "epoch","loss", "time"]).sort_values('time').groupby("epoch").last()
dfchkpt = dfchkpt.reset_index().sort_values('epoch').set_index("filename")
#dfchkpt.plot(x='epoch', y='loss')
WEIGHTFILE = dfchkpt["loss"].argmin()
# WEIGHTFILE = "./checkpoints/1cfdc49901cec8c56b66fb200de72f57/model.07-0.055612.hdf5"
print(WEIGHTFILE)

with open(os.path.join(indir, "checkpoint.info")) as chkpt_fh:
    prms = yaml.load(chkpt_fh)
    print("\n".join(["%s\t%s" %(kk,vv) for kk,vv in prms.items()]),)
##############################################3
fnmeta = "/data/dlituiev/tables/2017-06-mammo_tables/df_dcm_reports_birads_path_indic_dens_birad_wi_year_noreport_nodupl.csv.gz"


df = pd.read_csv(fnmeta)

pngdir = "/media/exx/tron/2017-07-png-jae/"
df["png"] = df["id"].map(lambda x: os.path.join(pngdir, x+".png"))


#model = get_model(n_classes=prms["n_classes"],
#              final_activation=prms["final_activation"],
#              ndense=prms["ndense"],
#              base_trainable=prms["base_trainable"])

if WEIGHTFILE:
    print("loading weights from:\t%s" % WEIGHTFILE)
    model = load_model(WEIGHTFILE)

prms["loss"] = '{}_crossentropy'.format( prms["class_mode"] )
model.compile(optimizer=Adam(lr=prms["lr"]), loss=prms["loss"], metrics=['accuracy'])
    

fliplr = True
if fliplr:
    preprocessing_function = lambda x: x[...,::-1,:]
    flipsuffix = 'fliplr'
else:
    preprocessing_function = None
    flipsuffix = 'orig'

fnbase = os.path.basename(fnmeta).replace(".gz","").replace(".csv","")
fnoutpred = os.path.join(os.path.dirname(fnmeta),
                         '{}-spotmag_img_prediction-{}-{}.csv'.format(
                             fnbase, indir.split('/')[1],
                             flipsuffix))

print("SAVING TO", fnoutpred)
try:
    os.unlink(fnoutpred)
except:
    print("%s\tnot found" % fnoutpred)
    pass

biter = batch_iterator(df["png"].tolist(), batch_size=batch_size,
                       preprocessing_function=preprocessing_function)

for nn, (filenames_, batch) in enumerate(biter):
    yscore = model.predict(batch)
    index = [ff.split("/")[-1].replace(".png","") for ff in filenames_]
    dfout = pd.Series(yscore.ravel(), 
            index=index)
    dfout.to_csv(fnoutpred, mode='a', header=None)
    print(nn)

print("DONE")
