# coding: utf-8
import numpy as np
import pandas as pd
import dicom
from warnings import warn

def get_tuples(plan, outlist = None, key = ""):
    if len(key)>0:
        key =  key + "_"
    if not outlist:
        outlist = []
    for aa  in plan.dir():
        if (hasattr(plan, aa) and aa!='PixelData'):
            value = getattr(plan, aa)
            if type(value) is dicom.sequence.Sequence:
#                 if len(list(value))==1:
#                     outlist.extend(get_tuples(list(value)[0], outlist = None, key = key+aa))
#                 else:
                for nn, ss in enumerate(list(value)):
                    newkey = "_".join([key,("%d"%nn),aa]) if len(key) else "_".join([("%d"%nn),aa])
                    outlist.extend(get_tuples(ss, outlist = None, key = newkey))
            else:
                if type(value) is dicom.valuerep.DSfloat:
                    value = float(value)
                elif type(value) is dicom.valuerep.IS:
                    value = str(value)
                elif type(value) is dicom.valuerep.MultiValue:
                    value = tuple(value)
                elif type(value) is dicom.UID.UID:
                    value = str(value)
                outlist.append((key + aa, value))
    return outlist


def filter_row_common_field(row, common_fields):
    for kk in list(row.keys()):
        if kk not in common_fields:
            row.pop(kk)
    return row



"""
fn_allheaders = '/home/dlituiev/data_dlituiev/manuallabeller/filelist/dicom_headers_all_fields_filelist_nonscreening_4000_seed42.csv'

df_allheaders = pd.read_csv(fn_allheaders, index_col=0)


"at least 5% of rows are there"
thr = 0.05
valid_fields = (~df_allheaders.isnull()).mean() > thr
valid_fields = valid_fields[valid_fields].index.tolist()
print(len(valid_fields))
"""

valid_fields = pd.read_table("/data/dlituiev/learn_spotmag_from_dicom_headers/LogisticRegression_common_fields_names.tab", 
                             header=None,
                            squeeze=True).values


#filelist_fn = '/home/dlituiev/data_dlituiev/tables/df_newest_mammos.pickle'
filelist_fn = "/home/dlituiev/data_dlituiev/tables/2017-06-mammo_tables/df_original_mammos.pickle"
filelist = pd.read_pickle(filelist_fn, )["Filename"].unique().tolist()
len(filelist)

BUFFER_N_LINES = 100
SEP = '\t'
outpath = filelist_fn.replace('.pickle','') + '_dicom_headers_selected.tab'
final_columns = ['filename'] + list(valid_fields)
print("len(final_columns)", len(final_columns) )
print('saving to %s' % outpath)
with open(outpath, 'w+') as outfh:
    outfh.write(SEP.join(final_columns) + '\n')
    headerlist = []
    for nn, ff in enumerate(filelist):
        if nn% BUFFER_N_LINES == (BUFFER_N_LINES-1):
            df_hl = pd.DataFrame( headerlist, columns=final_columns)
            df_hl.to_csv(outfh, sep=SEP, header=None, index=None, mode = 'a')
            outfh.flush()
            del df_hl
            print(nn+1)
            headerlist = []
        try:
            plan = dicom.read_file(ff)
            row = get_tuples(plan)
            row = dict(row)
            row = tuple([ff] + [(row[kk] if (kk in row) else np.nan) for kk in valid_fields ])
            print("len(row)", len(row))
            headerlist.append(row)
        except Exception as ex:
#             raise ex
            warn('header extraction failed on #\t%s\t%s\t%s' % (nn, ff, ex))
    # in the end, print the rest:
    df_hl = pd.DataFrame( headerlist, columns=final_columns)
    df_hl.to_csv(outfh, sep=SEP, header=None, index=None, mode = 'a')
    outfh.flush()

print("DONE")
