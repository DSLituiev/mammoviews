
# coding: utf-8

#cell#

import pandas as pd
import sys
from header_cleaner import get_features, normalize_fields, parse_float_tuples, parse_float

#cell#
fn_features = "../tables/df_all_mammos_dicom_headers_selected.tab.gz"
outfn = "../tables/df_all_mammos_dicom_headers_selected_norm.tab"

dffeatures = pd.read_table(fn_features, index_col="filename")

#cell#
mask_nonnumeric = ~dffeatures["ContentTime"].map(lambda x: isinstance(x, float) | isinstance(x, int))
dffeatures.loc[mask_nonnumeric, "ContentTime"] = dffeatures["ContentTime"][mask_nonnumeric].map(lambda x: float(x.replace(':','').replace('--',"30")))

#cell#
print("shape", dffeatures.shape)

#cell#
normalize_fun = {"0_ViewCodeSequence__0_ViewModifierCodeSequence_CodeMeaning":
                lambda x: str(x).lower(),
                "0_ViewCodeSequence_CodeValue": lambda x: str(x),
                "Grid": lambda x: str(x).replace("'","")
                                       .replace("(","").replace(")","")
                                       .replace(",","").replace("/"," ")
                                       .replace('PARRALLEL',"PARALLEL")
                                       .lower(),
                "HighBit": lambda x: str(int(x)) if (isinstance(x, float) and x*1==x) else str(x),
                "WindowCenter": lambda x: np.median(parse_float_tuples(x)),
                "FieldOfViewOrigin":parse_float_tuples,
                "EstimatedRadiographicMagnificationFactor": lambda x: x,
                "ContentTime": lambda x: x,
                "FieldOfViewRotation": lambda x: float(parse_float(x)),
                "KVP": lambda x: float(parse_float(x)),
                 "ShutterLowerHorizontalEdge":  lambda x: float(parse_float(x)),
                 "ShutterRightVerticalEdge":   lambda x: float(parse_float(x)),
                 "XRayTubeCurrentInuA": lambda x: float(parse_float(x)),
                 "RelativeXRayExposure": lambda x: float(parse_float(x)),
                 "ManufacturerModelName": lambda x: str(x).lower().replace('"',''),
                 "Manufacturer": lambda x: str(x).lower().replace('"','').replace(',', '').replace(" inc", "").rstrip('.'),
                 "BodyPartThickness":lambda x: float(parse_float(x)),
                 "CollimatorLeftVerticalEdge": lambda x: float(parse_float(x)),
                 "CollimatorLowerHorizontalEdge": lambda x: float(parse_float(x)),
                 "DetectorActiveDimensions" : lambda x: parse_float_tuples(x.replace("\\", ", ") if isinstance(x, str) else x),
                 "ExposureTime": lambda x: x,
                 "ExposuresOnDetectorSinceLastCalibration": lambda x: x,
                 "ExposuresOnDetectorSinceManufactured": lambda x: x,
                 "DistanceSourceToEntrance":  lambda x: x,
                 "DetectorTemperature":lambda x: float(parse_float(x)),
                 "DistanceSourceToDetector":  lambda x: x,
                 
}

dtypes = {"0_ViewCodeSequence__0_ViewModifierCodeSequence_CodeMeaning": str,
                "0_ViewCodeSequence_CodeValue": str,
                "Grid": str,
                "HighBit": str, # int
                "WindowCenter": int,
                "FieldOfViewOrigin": 'O',
                "EstimatedRadiographicMagnificationFactor": float,
                "ContentTime": float, #NaN
                "FieldOfViewRotation": float,
                "KVP": float,
                 "ShutterLowerHorizontalEdge": float,
                 "ShutterRightVerticalEdge": float,
                 "XRayTubeCurrentInuA": float,
                 "RelativeXRayExposure": float,
                 "ManufacturerModelName": str,
                 "Manufacturer": str,
                 "BodyPartThickness": float,
                 "CollimatorLeftVerticalEdge": float,
                 "CollimatorLowerHorizontalEdge": float,
                 "DetectorActiveDimensions" : 'O',
                 "ExposureTime": float,
                 "ExposuresOnDetectorSinceLastCalibration": float, # NaNs
                 "ExposuresOnDetectorSinceManufactured": float, # NaNs
                 "DistanceSourceToEntrance": float,
                 "DetectorTemperature": float,
                 "DistanceSourceToDetector": float,
                 
}

#cell#

set(dffeatures.columns) - set(normalize_fun.keys())

#cell#

for kk, vv in dffeatures.items():
    print(kk)
    dffeatures.loc[:,kk] = vv.map(normalize_fun[kk]).astype(dtypes[kk])

dffeatures.to_csv(outfn, sep='\t',  compression='gzip')
