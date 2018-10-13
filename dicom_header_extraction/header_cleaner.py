
# coding: utf-8

import numpy as np
import pandas as pd
import os
from functools import partial
from itertools import chain

def entropy(x):
    f = x.value_counts()
#     f.loc["nan"] = x.isnull().sum()
    return (f*f.map(np.log2)).sum()


def select_text_fields(df_allheaders):
    text_fields = df_allheaders.dtypes.map(lambda x: x is pd.np.dtype(object))
    text_fields = text_fields[text_fields].index.tolist()
    len(text_fields)
    text_fields = (~df_allheaders[text_fields].isnull()).mean() > 0.05

    text_fields = text_fields[text_fields].index.tolist()
    remove_list = []
    for tt in text_fields:
        numunique = len(df_allheaders[tt].unique())
        entr = entropy(df_allheaders[tt])
        if entr<1000 | (numunique == 1) | (numunique > 0.75*df_allheaders.shape[1]):
            remove_list.append(tt)
    
    for tt in remove_list:
        text_fields.remove(tt)

    len(text_fields)
    return text_fields


def get_good_numeric_fields(df_allheaders, thr_stderr = 1e-6):
    stderr = df_allheaders.std()/df_allheaders.mean()
    field_list = stderr[stderr> thr_stderr].index.tolist()
    return field_list


def get_index_from_int_tuple(x, ind):
    if type(x) is str:
        x = eval(x)
        return int(float(x[ind]))
    else:
        return x


def clean_up_field_list(field_list, 
     prefices_remove = ["date", "accession", "number", 
         "Filename",
         "ImageLaterality",
         "GantryID",
         #"0_ViewCodeSequence_CodeMeaning",
         "ViewCodeSequence_CodeMeaning",
         "ViewModifierCodeSequence_CodeValue",
         "EthnicGroup",
         "BodyPartExamined",
         "LossyImageCompression",
         "DeidentificationMethodCodeSequence",
         "UID",
         'EntranceDoseInmGy',
         'ProcedureCodeSequence_CodeMeaning',
         'CommentsOnRadiationDose',
         'DetectorID',
         'SeriesDescription', # potentially informative but too many values
         'SoftwareVersions',
         'PatientAge',
         ],
     fields_remove = [ 'PatientID', 'PatientName', "BitsStored",
         'AcquisitionTime', 
         'AdmittingTime', 
         'ScheduledStudyStartTime',
         'InstanceCreationTime',
         'PerformedProcedureStepStartTime',
         'PregnancyStatus',
         'StudyArrivalTime',
         'StudyCompletionTime',
         'StudyTime',
         'TimeOfLastCalibration',
         'TimeOfLastDetectorCalibration',
         'TimeOfSecondaryCapture',]):

    prefices_remove = [x.lower() for x in prefices_remove]

    for ff in field_list:
        for pp in prefices_remove:
            if pp in ff.lower():
                if ff not in fields_remove:
                    fields_remove.append(ff)

    for ff in fields_remove:
        try:
            field_list.remove(ff)
        except ValueError as ve:
            print(ff, ve)
    return field_list


def make_lowercase_text_fields(df_allheaders):
    """## make all text fields lowercase 
    (except accession and file name)"""
    for cname in df_allheaders.columns[1:]:
        cc = df_allheaders[cname]
        if cc.dtype is np.dtype(object):
            df_allheaders[cname] = cc.str.lower()
    return df_allheaders


def format_PixelSpacing(x):
    if type(x) is float:
        return x
    else:
        xstr = x.lstrip("(").rstrip(")").replace("'", "").replace(" ","").split(",")
        return np.unique(tuple([float(y) for y in xstr]))[0]

def parse_float(x):
    x = str(x).replace("'","").replace("b","").replace("None","nan")
    if x == "":
        x = np.nan
    return x

def parse_float_tuples(x, to_int=False):
    x = list(str(x))
    for nn,ss in enumerate(x):
        if not ss.isdigit() and ss!='.':
            x[nn] = ';'
    x = "".join(x).split(';')
    if to_int:
        x = tuple([int(float(dd)) for dd in x if len(dd)])
    else:
        x = tuple([float(dd) for dd in x if len(dd)])
    if type(x) is not tuple:
        raise TypeError("returned non-list: {}".format(str(x)))
    return x

def parse_float_tuples_prod(x):
    if x not in (None, np.nan) and len(x)>0:
        x = str(x)
        assert type(x) is str
        x = parse_float_tuples(x)
        if type(x) is not tuple:
            raise TypeError("returned non-list: {} of type {}".format(str(x), type(x)))
        try:
            x = np.prod(x)
        except TypeError as ee:
            print('"%s"' % x)
            raise ee
    else:
        x = np.nan
    return x

def parse_int_tuples_median(x):
    x = parse_float_tuples(x)
    x = np.median(x)
    return x
"""
def parse_float_tuples(x):
    x = eval(x) if type(x) is str else x
    if type(x) in [tuple, list]:
        x = tuple([float(y) for y in x])
    return x
"""

def parse_str_tuples(x):
    try:
        x = eval(x) if type(x) is str else x
    except:
        x = tuple(x.split(" ")) if type(x) is str else x
    return x
#############################33
def extract_list_text_field(df_allheaders, colprefix = "ViewModifierCodeSequence_CodeMeaning"):
    allcols = df_allheaders.columns
    cols = allcols[np.asarray(allcols.map(lambda x: colprefix in x and x!=colprefix), dtype=bool)]

    ViewModifierCodeSequence_CodeMeaning = set()
    for cc in cols:
        ViewModifierCodeSequence_CodeMeaning |= set(df_allheaders[cc].dropna().unique())

    for vv in (True, False):
        if (vv in ViewModifierCodeSequence_CodeMeaning):
            ViewModifierCodeSequence_CodeMeaning.remove(vv)
        
    ViewModifierCodeSequence_CodeMeaning = dict(zip(
            ViewModifierCodeSequence_CodeMeaning,
           [None]*len(ViewModifierCodeSequence_CodeMeaning)))
    
    for kk in ViewModifierCodeSequence_CodeMeaning.keys():
        ViewModifierCodeSequence_CodeMeaning[kk] = df_allheaders[cols[0]].copy()
        ViewModifierCodeSequence_CodeMeaning[kk][:] = False
        ViewModifierCodeSequence_CodeMeaning[kk] = \
            ViewModifierCodeSequence_CodeMeaning[kk].astype(bool)
        for cc in cols:
            ViewModifierCodeSequence_CodeMeaning[kk] |= df_allheaders[cc].map(lambda x: kk in x if type(x) is str else False) 


    ViewModifierCodeSequence_CodeMeaning = pd.DataFrame(ViewModifierCodeSequence_CodeMeaning)
    ViewModifierCodeSequence_CodeMeaning.columns = \
        ViewModifierCodeSequence_CodeMeaning.columns.map(lambda x: colprefix + "_" + x.replace(" ",""))
    
    for cc in cols:
        df_allheaders.drop(cc, axis=1, inplace=True)
    df_allheaders = pd.concat([df_allheaders, ViewModifierCodeSequence_CodeMeaning], axis=1)
    return df_allheaders

#############################33
def normalize_fields(df_allheaders):
    # ## Clean up
    # ### PixelSpacing
    if "PatientAge" in df_allheaders.columns:
        df_allheaders.PatientAge = df_allheaders.PatientAge.map(lambda x: int(x.lower().rstrip('y')))
    if "DetectorActiveDimensions" in  df_allheaders.columns:
        df_allheaders.DetectorActiveDimensions = df_allheaders.DetectorActiveDimensions.map(parse_float_tuples_prod)
        #df_allheaders.DetectorActiveDimensions = list(map(parse_float_tuples_prod,
        #                                df_allheaders.DetectorActiveDimensions.tolist()))

    if "PixelSpacing" in  df_allheaders.columns:
        df_allheaders.PixelSpacing = df_allheaders["PixelSpacing"].map(format_PixelSpacing)
    if "ImagerPixelSpacing" in df_allheaders.columns:
        df_allheaders.ImagerPixelSpacing = df_allheaders["ImagerPixelSpacing"].map(format_PixelSpacing)
    if "ModalitiesInStudy" in df_allheaders.columns:
        df_allheaders["ModalitiesInStudy"] = df_allheaders["ModalitiesInStudy"].map(lambda x: "mg" in str(x))
    if "HalfValueLayer" in df_allheaders.columns:
        df_allheaders["HalfValueLayer"] = df_allheaders["HalfValueLayer"].map(lambda x: x if type(x) is float else float(str(x).replace('b','').replace("'", '')))
    


    # ### FieldOfViewDimensions
    # computing area and filling in the gaps with the mode **worsens** the FNR

    # df_allheaders['FieldOfViewDimensions'] = df_allheaders['FieldOfViewDimensions'].map(lambda x: np.prod([int(y) for y in eval(x)]) if type(x) is str else x)
    # df_allheaders.loc[df_allheaders['FieldOfViewDimensions'].isnull(), 'FieldOfViewDimensions'] = df_allheaders['FieldOfViewDimensions'].value_counts().argmax()


    # df_allheaders["PartialView"].map(lambda x: type(x)).value_counts()
    if "ViewPosition" in df_allheaders.columns:
        df_allheaders["ViewPosition"] = df_allheaders["ViewPosition"].map(lambda x: x in ['cc', 'mlo'])

    df_allheaders = extract_list_text_field(df_allheaders, 
        colprefix = "ViewModifierCodeSequence_CodeMeaning")

    #df_allheaders = extract_list_text_field(df_allheaders, 
    #    colprefix = "ViewModifierCodeSequence_CodeMeaning")

    # ### BreastImplantPresent
    # #### clean up
    if "BreastImplantPresent" in df_allheaders.columns:
        # BreastImplantPresent = pd.Series([np.nan]*df_allheaders.shape[0])
        #BreastImplantPresent = pd.Series([False]*df_allheaders.shape[0])
        #BreastImplantPresent[df_allheaders["BreastImplantPresent"].map(str).map(lambda x: "yes" in x)] = True
        BreastImplantPresent = df_allheaders["BreastImplantPresent"].map(str).map(lambda x: "yes" in x)
        # BreastImplantPresent[df_allheaders["BreastImplantPresent"].map(str).map(lambda x: "no" in x)] = False
        df_allheaders['BreastImplantPresent'] = BreastImplantPresent
        del BreastImplantPresent
    if "PartialView" in df_allheaders:
        df_allheaders["PartialView"] = df_allheaders["PartialView"].map(lambda x : "yes" in x if type(x) is str else False)

    for kk in ["WindowWidth", "WindowCenter"]:
        if kk in df_allheaders.columns:
            df_allheaders[kk] = df_allheaders[kk].map(parse_int_tuples_median)
    
    if "PatientOrientation" in df_allheaders.columns:
        df_allheaders.PatientOrientation = df_allheaders.PatientOrientation.map(parse_str_tuples)
    if "DetectorElementPhysicalSize" in df_allheaders.columns:
        df_allheaders["DetectorElementPhysicalSize"] = df_allheaders.DetectorElementPhysicalSize.map(parse_float_tuples)
    # ### Grid
    # df_allheaders["Grid"].value_counts()
    if "Grid" in df_allheaders.columns:
        df_allheaders["Grid"] = (df_allheaders["Grid"]
                             .map(str)
                             .map(lambda x: x.replace('(','')
                                             .replace(')','')
                                             .replace("'","")
                                             .replace(',','')
                                             .replace("parrallel", "parallel")))

        df_allheaders.loc[df_allheaders["Grid"] == "('reciprocating', 'parrallel')", "Grid"] = "('reciprocating', 'parallel')"
        df_allheaders["Grid"].value_counts()
    # df_allheaders.PixelSpacing = df_allheaders.PixelSpacing.astype(str)
    # df_allheaders.PixelSpacing.value_counts()
    if "FieldOfViewOrigin" in df_allheaders.columns:
        df_allheaders["FieldOfViewOrigin_x"] = df_allheaders.FieldOfViewOrigin.map(lambda x : get_index_from_int_tuple(x, 0))
        df_allheaders["FieldOfViewOrigin_y"] = df_allheaders.FieldOfViewOrigin.map(lambda x : get_index_from_int_tuple(x, 1))
        df_allheaders.drop("FieldOfViewOrigin", axis=1, inplace=True)

    #informative_cols.remove("FieldOfViewOrigin")
    #informative_cols.append("FieldOfViewOrigin_x")
    #informative_cols.append("FieldOfViewOrigin_y")
    if "FocalSpots" in df_allheaders.columns: 
        df_allheaders.loc[df_allheaders["FocalSpots"].isnull(), "FocalSpots"] = df_allheaders["FocalSpots"].value_counts().argmax()
    for kk in ["PixelSpacing", "EstimatedRadiographicMagnificationFactor", "XRayTubeCurrent", "DistanceSourceToPatient"]:
    #    print(kk)
        if kk in df_allheaders.columns:
            df_allheaders.loc[df_allheaders[kk].isnull(), kk] = df_allheaders[kk].median()
    if "ImageType" in df_allheaders.columns:
        keywords = set(chain(*(df_allheaders.ImageType.map(lambda x: parse_str_tuples(x)).tolist())))
        keywords.remove("")
        for kk in  keywords:
            key = "ImageType"+"_"+kk
            df_allheaders[key] = df_allheaders.ImageType.map(lambda x: kk in x)
        df_allheaders.drop("ImageType", axis=1, inplace=True)

    return df_allheaders


def move_digits_back(allcolumns):
    allcolumns = list(allcolumns)
    for nn, x in enumerate(allcolumns):
        if x[0] in set(list('0123456789')):
            x = "_".join(x.split("_")[1:] + x.split("_")[:1])
            allcolumns[nn] = x
    return allcolumns

def get_features(df_allheaders, thr_stderr = 1e-6):
    # df_allheaders.columns = move_digits_back(df_allheaders.columns)

    df_allheaders = normalize_fields(df_allheaders.copy())
    text_fields = select_text_fields(df_allheaders)
    # df_allheaders[text_fields].apply(entropy).hist()

    if  thr_stderr >0:
        field_list = get_good_numeric_fields(df_allheaders,thr_stderr=thr_stderr)
    field_list = list(set(clean_up_field_list(field_list + text_fields)))

    df_allheaders = make_lowercase_text_fields(df_allheaders)

    # pd.crosstab(df_allheaders['0_ViewCodeSequence_CodeMeaning'], df_allheaders['ViewPosition'])
    # informative_cols = ['Filename', 'AccessionNumber','BreastImplantPresent','DistanceSourceToPatient','EstimatedRadiographicMagnificationFactor',
    #                  'FocalSpots','Grid','PixelSpacing','XRayTubeCurrent', 'ViewPosition', 'PartialView']

    informative_cols = ['Filename', 'AccessionNumber'] + field_list

    feature_columns = informative_cols[2:]

    noncategorical = ['ContentTime',
                     'FieldOfViewOrigin_x',
                     'FieldOfViewOrigin_y',
                     'HalfValueLayer',
                     'WindowWidth',
                     'CompressionForce',
                    'DetectorActiveDimensions',
                    'RelativeXRayExposure',
                    'ExposureTime',
                    'Exposure',
                    'BodyPartThickness',
                    'FieldOfViewOrigin_y',
                    'CollimatorLowerHorizontalEdge',
                    'WindowCenter',
                    'FieldOfViewRotation',
                    'KVP',
                    'DistanceSourceToDetector',
                    'DistanceSourceToEntrance',
                    'CollimatorLeftVerticalEdge',
                    'DetectorTemperature',
                    'HighBit'] 
    categorical = ['Manufacturer',
                    'ManufacturerModelName',
                    'Grid_htc',
                    'ViewModifierCodeSequence_CodeMeaning',
                    'ViewModifierCodeSequence_CodeMeaning']

    noncategorical = list(set(feature_columns) & set(noncategorical))
    potentially_categorical = (set(feature_columns) - set(noncategorical))
    potentially_categorical |= set(categorical) & set(df_allheaders.columns)
    potentially_categorical = list(potentially_categorical)
    print("potentially_categorical", len(potentially_categorical))
    print("non_categorical", len(noncategorical))
    for cc in noncategorical:
        if str(df_allheaders[cc].dtype) == 'object':
            df_allheaders[cc] = df_allheaders[cc].map(parse_float).astype(float)
    if len(potentially_categorical)>0:
        df_allheaders[potentially_categorical] = df_allheaders[potentially_categorical].fillna('unknown')
        features_onehot = pd.get_dummies(df_allheaders[potentially_categorical], 
                            drop_first=True, prefix_sep='=')
        features_onehot = pd.concat([features_onehot, df_allheaders[noncategorical]], axis=1) 
    else:
        print("no features to binarise!")
        features_onehot = df_allheaders[non_categorical].copy()

    #features_onehot = pd.concat([df_allheaders.Filename, features_onehot],axis=1,).set_index("Filename")

    features_onehot.shape, features_onehot.dropna().shape

    # ### Map DICOM  file name to PNG file name (remove directories)
    #features_onehot.index = features_onehot.index.map(lambda x: "_".join(x.split("/")[-4:]).replace(".dcm", ".png")).tolist()
    for cc in features_onehot.columns[features_onehot.isnull().any()]:
        print("filling in with median:\t%s" % cc)
        features_onehot.loc[features_onehot[cc].isnull(),cc] = \
                features_onehot[cc].median()
    features_onehot = features_onehot.loc[:,~features_onehot.isnull().any()]

    onehotcols = np.asarray(features_onehot.columns[features_onehot.dtypes.map(lambda x : x is pd.np.dtype("uint8"))].tolist())
    thr_frac = 0.01
    bad_feature_cols = onehotcols[(features_onehot[onehotcols].sum(0) < 5) |
                                  (features_onehot[onehotcols].mean(0) < thr_frac) |
                                  (features_onehot[onehotcols].mean(0) > (1-thr_frac))]
    len(bad_feature_cols)
    features_onehot.drop(bad_feature_cols, axis=1, inplace=True)
    if "FocalSpots" in features_onehot:
        features_onehot.loc[features_onehot["FocalSpots"].isnull(), "FocalSpots"] = \
                features_onehot["FocalSpots"].value_counts().argmax()

    return features_onehot


#############################
if __name__ == '__main__':
    PREFIX="allfeatures"

    # !sudo pip3 install dicom 
    # # read a table of DICOM headers
    filelist_fn = '/home/dlituiev/data_dlituiev/manuallabeller/filelist/filelist_nonscreening_4000_seed42.csv'
    outpath = os.path.join(os.path.dirname(filelist_fn), "dicom_headers_all_fields_" + os.path.basename(filelist_fn))
    print(outpath)
    df_allheaders = pd.read_csv(outpath, index_col=0)
    features_onehot = get_features(df_allheaders)

    # ## Read labels
    fn_man_labels = "/data/dlituiev/tables/cleaned_manual_labels_valset_4000.txt"
    df = pd.read_table(fn_man_labels, index_col=0)
    df.index = df.index.map(lambda x : x.split("/")[-1])

    # process labels
    df["special_view"] = df["regular_view"].map(lambda x: not x)


    dfm = pd.merge(df[["special_view"]], features_onehot, how='left', left_index=True, right_index=True)
    dfm.shape

    import seaborn as sns
    import matplotlib.pyplot as plt
    from statsmodels.graphics.mosaicplot import mosaic
    plt.matplotlib.rcParams["hatch.color"] = [0.7]*3

    dfm.var()
    dfm.isnull().sum()
    dfm.plot(x='special_view', y='XRayTubeCurrent', kind='scatter', alpha=0.05)
    dfm.plot(x='special_view', y='DistanceSourceToPatient', kind='scatter', alpha=0.05)
    dfm["special_view"].isnull().sum()


    target = dfm["special_view"]
    features = dfm.drop("special_view", axis=1)


    from sklearn.utils import shuffle
    # for building and visualizing the decision tree
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    # from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    # visualization
    from vis_tree import visualize_tree
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                                 precision_score, roc_curve, precision_recall_curve)




    y_dev,  y_val, X_dev, X_val = train_test_split(target, features, random_state=0, test_size=1/6)

    y_tr,  y_ts, X_tr, X_ts = train_test_split(y_dev, X_dev, random_state=0, test_size=1/5)




    # dtree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, criterion="entropy")
    # dtree = RandomForestClassifier(min_samples_split=10, min_samples_leaf=5)
    # dtree = AdaBoostClassifier(base_estimator=dtree, n_estimators=60, learning_rate=0.01)
    # dtree = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=50, learning_rate=0.01)




    dtree = GradientBoostingClassifier(max_depth=8, n_estimators=40, learning_rate=0.05, min_samples_leaf=12)
    modelname = str((dtree).__class__).split(".")[-1].rstrip(""" "'> """).lstrip('"')




    dtree.fit(X_tr, y_tr)
    pred_y_ts = dtree.predict(X_ts)
    pred_yscore_ts = dtree.predict_proba(X_ts)




    get_ipython().magic('pinfo auc')




    pr_, rec_, thresholds = precision_recall_curve(y_ts.tolist(), pred_yscore_ts[:,1], pos_label=1)
    # auc_pr = auc(pr_, rec_)

    plt.plot(pr_, rec_)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    # plt.title('auPRC = {0:.2f}%'.format(auc_pr))
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.axis('equal')
    plt.axis('square')

    print("%.2f" % (100*auc_))
    frmt = 'png'
    plt.savefig("{}_{}_auc.{}".format(PREFIX, modelname, frmt), dpi=300, format=frmt)




    fpr_, tpr_, thresholds = roc_curve(y_ts.tolist(), pred_yscore_ts[:,1], pos_label=1)
    auc_ = auc(fnr_, tpr_)

    plt.plot(fpr_, tpr_)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC = {0:.2f}%'.format(auc_))
    plt.axis('equal')
    plt.axis('square')

    print("%.2f" % (100*auc_))
    frmt = 'png'
    plt.savefig("{}_{}_auc.{}".format(PREFIX, modelname, frmt), dpi=300, format=frmt)




    # pd.DataFrame(dict(FNR=fnr_, TPR=tpr_, threshold=thresholds))
    features.plot(x="EstimatedRadiographicMagnificationFactor", y="PixelSpacing", kind='scatter')




    fig,ax = plt.subplots(1, figsize=(6,14))
    feat_imp = pd.Series(dtree.feature_importances_, index=features.columns)
    feat_imp = feat_imp[feat_imp>0.0].sort_values()[::-1]
    feat_imp[::-1].plot(kind='barh', ax=ax)
    print(feat_imp)
    # plt.xlim([0,0.5])
    # plt.tight_layout()
    frmt = 'png'
    plt.savefig("{}_{}_feature_importances.{}".format(PREFIX, modelname, frmt), dpi=300, format=frmt)









    len(thresholds)




    # pd.DataFrame(dict(
    #     FNR=fnr_,
    #     TPR=tpr_,
    #     threshold = thresholds))




    df_confusion = pd.crosstab(pd.Series(y_ts.as_matrix(), name="observed"), pd.Series(pred_y_ts, name="predicted"))
    df_confusion
    confusion_matrix(y_ts, pred_y_ts)
    cm = confusion_matrix(y_ts, pred_y_ts)
    cm[1,0]/cm[1,:].sum()
    def fnr(dtree, X_val, y_val, thr = None):
        if not thr:
            pred_y_val = dtree.predict(X_val)
        else:
            pred_y_val = dtree.predict_proba(X_val)[:,1] > thr
    #     df_confusion = pd.crosstab(pd.Series(np.asarray(y_val), name="observed"),
    #                                pd.Series(pred_y_val, name="predicted"))
    #     out = df_confusion[False][True] / (df_confusion[False][True] + df_confusion[True][True])
        
        cm = confusion_matrix(y_val, pred_y_val)
        out = cm[1,0]/cm[1,:].sum()
        return out




    def fpr(dtree, X_val, y_val, thr = None):
        if not thr:
            pred_y_val = dtree.predict(X_val)
        else:
            pred_y_val = dtree.predict_proba(X_val)[:,1] > thr
    #     df_confusion = pd.crosstab(pd.Series(np.asarray(y_val), name="observed"),
    #                                pd.Series(pred_y_val, name="predicted"))
    #     out = df_confusion[True][False] / (df_confusion[False][False] + df_confusion[True][False])
        
        
        cm = confusion_matrix(y_val, pred_y_val)
        if cm[0,:].sum() !=0:
            out = cm[0,1]/cm[0,:].sum()
        else:
            out = 0.0
        return out









    THR = 0.15


    #          True | False
    #     True   TP |  FN
    #     False  FP |  TN
    # 
    # 
    #     FPR = FP / (FP + TN)
    # 



    pred_y_ts = dtree.predict_proba(X_ts)[:,1] > THR
    df_confusion = pd.crosstab(pd.Series(y_ts.as_matrix(), name="observed"), pd.Series(pred_y_ts, name="predicted"))
    print(df_confusion.to_csv(sep='|'))




    THR = 0.05

    modelname = str((dtree).__class__).split(".")[-1].rstrip(""" "'> """).lstrip('"')
    cv_fnr = cross_val_score(dtree, X_dev, y_dev, groups=None, scoring=partial(fnr, thr=THR), cv=5, n_jobs=1, pre_dispatch='2*n_jobs')
    cv_fpr = cross_val_score(dtree, X_dev, y_dev, groups=None, scoring=partial(fpr, thr=THR), cv=5, n_jobs=1, pre_dispatch='2*n_jobs')

    tmpstr = """model: {}
    threshold = {}
    + on the hold-out set:\tFNR = {:.2f}%, FPR = {:.2f}%
    + in 5-fold cross-validation (mean):\tFNR = {:.2f}%, FPR = {:.2f}%""".format(
        modelname, THR, 
        100*fnr(dtree, X_ts, y_ts, thr = THR), 100*fpr(dtree, X_ts, y_ts, thr = THR),
        100*cv_fnr.mean(), 100*cv_fpr.mean())
    print(tmpstr)




    THR = 0.5
    modelname = str((dtree).__class__).split(".")[-1].rstrip(""" "'> """).lstrip('"')
    cv_fnr = cross_val_score(dtree, X_dev, y_dev, groups=None, scoring=partial(fnr, thr=THR), cv=5, n_jobs=1, pre_dispatch='2*n_jobs')
    cv_fpr = cross_val_score(dtree, X_dev, y_dev, groups=None, scoring=partial(fpr, thr=THR), cv=5, n_jobs=1, pre_dispatch='2*n_jobs')

    tmpstr = """model: {}
    threshold = {}
    + on the hold-out set:\tFNR = {:.2f}%, FPR = {:.2f}%
    + in 5-fold cross-validation (mean):\tFNR = {:.2f}%, FPR = {:.2f}%""".format(
        modelname, THR, 
        100*fnr(dtree, X_ts, y_ts, thr = THR), 100*fpr(dtree, X_ts, y_ts, thr = THR),
        100*cv_fnr.mean(), 100*cv_fpr.mean())
    print(tmpstr)




    6/72


    # ## fnr
    # 0.1443 -- AdaBoostClassifier(50, lr=0.1) with:
    # 
    # 
    #     DecisionTreeClassifier(max_depth=7, min_samples_leaf=5, criterion="entropy")
    #     GaussianNB()
    # 
    # 0.1134 -- AdaBoostClassifier(50, lr=0.01) with:
    #     GaussianNB()



    accuracy_score(y_true=y_val, y_pred=pred_y_val)




    f1_score(y_true=y_val, y_pred=pred_y_val)









    confusion_matrix(y_true=y_val, y_pred=pred_y_val)





    df_confusion = pd.crosstab(pd.Series(y_val.as_matrix(), name="observed"), 
                               pd.Series(pred_yscore_dev[:,1]>0.15, name="predicted"))
    df_confusion




    df_confusion[False][True] / (df_confusion[False][True] + df_confusion[True][True])




    df_confusion[True][False] / (df_confusion[False][False] + df_confusion[True][False])




    109/(385+109)


    # ## Misclassified: examples and comments



    # pred_false = (pd.Series(pred_y_val, name="predicted")==False)
    pred_false = (pd.Series(pred_yscore_dev[:,1]<0.15, name="predicted")==False)
    false_negatives = (pd.Series(y_val.as_matrix(), name="observed")) & pred_false
    false_negatives.index=y_val.index
    false_negatives.shape, df.shape
    # y_val[false_negatives.tolist()].shape 














    xstr = """1805162996_1.2.840.113654.2.70.1.75424722723272471565664976911416714890_2_37.png -- implant?
    1433463766_1.2.840.113654.2.70.1.243422935316700791950696878743366703411_6_6.png -- male?
    3395322213_1.2.840.113654.2.70.1.161905211577383187509354224390811944382_1161_7.png -- overexposed with scale grid
    1383662805_1.2.840.113654.2.70.1.194667288082835549565211946781626641146_1_88.png -- mag? bars in the image
    5717508670_1.2.840.113654.2.70.1.135196805563780165444562848954663016070_2_6.png -- spot
    1582554801_1.2.840.113654.2.70.1.202883517655342643705007475928329105895_1_1.png -- strange shape; plate
    3248534628_1.2.840.113654.2.70.1.153327658320065917717726871735320153117_14_8.png -- RLMID, implant
    1050998385_1.2.840.113654.2.70.1.294672228525412928579179278566440354700_168_12.png -- RMLO, underexposed, plate
    2431514667_1.2.840.113654.2.70.1.132697486450403983700631264913146412468_1_1.png -- regular CC
    2836025574_1.2.840.113654.2.70.1.94728406891527814842052605970255602447_31728_4.png  -- regular CC, wire?
    2774547752_1.2.840.113654.2.70.1.152335331945150793610356395498084601027_47428_6.png  -- poor exposure?
    6784971236_1.2.840.113654.2.70.1.276140387730485551768768734852859745761_21705_2.png  -- regular CC
    6120027884_1.2.840.113654.2.70.1.202389441802705593488291262945242015864_28128_3.png -- spot
    2127109953_1.2.840.113654.2.70.1.136443797025605972119376095795980286524_5_26.png  -- RML, scar
    5015120217_1.2.840.113654.2.70.1.8576402180164318136049174781190805706_19615_3.png -- regular MLO, underexposure
    2915273528_1.2.840.113654.2.70.1.50904067248781976561131370015339684052_3_51.png -- RLM
    2859796079_1.2.840.113654.2.70.1.248757700026158935826319533755178408586_3_51.png -- LMLO, scar""".split("\n")




    df_misclassified_comments = pd.DataFrame([x.split(" -- ") for x in xstr], columns=["Filename", "comment"]).applymap(lambda x: x.rstrip().lstrip()).set_index("Filename")["comment"]
    df_misclassified_comments




    df_misclassified_comments[false_negatives & X_val[false_negatives]['ViewPosition'] & ~X_val[false_negatives]['ViewModifierCodeSequence'] ]




    df_misclassified_comments[false_negatives & X_val[false_negatives]['ViewPosition'] & ~X_val[false_negatives]['ViewModifierCodeSequence'] ]




    X_val.columns




    # X_val[false_negatives][['ViewPosition_ccid', 'ViewPosition_lm', 'ViewPosition_lmid',
    #        'ViewPosition_ml', 'ViewPosition_mlo', 'ViewPosition_mloid',
    #        'ViewPosition_xccl', "FieldOfViewDimensions_('145', '105')"]]

    X_val[false_negatives][['ViewPosition', 
                           'ViewModifierCodeSequence']]

