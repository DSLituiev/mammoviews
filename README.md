# Code for automatic labeling of special diagnostic mammography views from images and DICOM headers

## DICOM
### Extract selected fields from DICOM headers

    dicom_header_extraction/extract_dicom_headers_w_generator_150K.py

### Normalize / expand data

    dicom_header_extraction/normalize_selected_dcm_headers.py

###  Machine learning on DICOM headers

    caret_on_headers.R       # most methods 
    caret_on_headers_nona.R  # GLMNET

## Image pipeline

### General image model
- scripts and config files: `image_classifiers/e5ce2d69b035975cb5336cec0da9a32a`

- weight files:

### Wire localization model

- scripts and config files: `image_classifiers/e8e71fc090141d7c6fb334359152d295`

- weight files:


## Visualization of performance metrics 
Scripts used to generate Fig. 1

    combine_predictions_hdr_and_img.ipynb
    visualize_predictions_hdr_and_img.ipynb


## Significance tests
Scripts used to generate Supplementary Figures S1 & S2

    calc_auroc_confidence_intervals.R
    plot_auroc_difference_pvalue.ipynb
