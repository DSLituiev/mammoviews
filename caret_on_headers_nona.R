# coding: utf-8
############################################################################
# stratify by BT column: those are 100% sure digital, others can be either
############################################################################
rm(list=ls())
setwd(dir = "~/repos/mammo/learn_spotmag_from_dicom_headers")
#cell#
library(caret)
library(data.table)

library(pROC)
# install.packages(c("pROC"))
library(ggplot2)
library(fastmatch)

read.gz <- function(filename, ...){
  as.data.frame(fread(paste("zcat < ",filename),
                      header=TRUE,  fill = TRUE, ...))
}


fn_ids = "../tables/2017-06-mammo_tables/df_dcm_reports_birads_path_indic_dens_birad_wi_year_noreport_nodupl.csv.gz"
ids = read.gz(fn_ids, select="id")$id

fn_features = "../tables/mammo_dicom_headers/df_all_mammos_dicom_headers_selected_nona.tab.gz"
dffeatures = read.gz(fn_features, sep='\t')

# rownames(dffeatures) <- dffeatures$filename
print(nrow(dffeatures))
print(length(ids))

dffeatures <- dffeatures[fmatch(unique(ids), dffeatures$filename),]
dffeatures <- dffeatures[!is.na(dffeatures$filename),]

rm(ids)

collist = c("BodyPartThickness", "XRayTubeCurrentInuA",  "ContentTime",
            "DetectorTemperature", "WindowCenter", "FieldOfViewRotation")
for (cc in collist){
  dffeatures[,cc] <- as.numeric(dffeatures[,cc])
}



# (head(as.numeric(dffeatures$BodyPartThickness)))
dtypes = sapply(dffeatures, class)

row.names(dffeatures) = dffeatures$filename
excludeCols <- c("filename",
                 "CollimatorLeftVerticalEdge",
                 "CollimatorLowerHorizontalEdge",
                 "DistanceSourceToEntrance",
                 "ExposuresOnDetectorSinceLastCalibration",
                 "ExposuresOnDetectorSinceManufactured",
                 "ShutterLowerHorizontalEdge",     
                 "ShutterRightVerticalEdge",
                 "XRayTubeCurrentInuA"
                 # "ManufacturerModelName"
)
dffeatures <- (dffeatures[, !(colnames(dffeatures) %in% excludeCols)])


catcols <- c('ViewModifierCodeMeaning',
             'ViewCodeValue',
             'DetectorActiveDimensionsMissing',
             'FieldOfViewOriginMissing',
             'Grid',
             'Manufacturer',
             'ManufacturerModelName')

for (cc in catcols){
  dffeatures[,cc] = paste0("=", dffeatures[,cc])
  dffeatures[,cc] = as.factor(dffeatures[,cc])
}

dffeatures[,"HighBit"] <- as.numeric(dffeatures[,"HighBit"])

colSums(sapply(dffeatures, is.na))

# Read labels ---------------------------------

fn.labelledset = "../tables/spotmag_predictions/train_test_split-2018-02-15-within7e5.csv"
# filelist.labelled = read.table(fn.labelledset, )
df.labelled = as.data.frame(fread(fn.labelledset))
rownames(df.labelled) <- df.labelled$id
vec.labelled = df.labelled$id
df.labelled$label <- as.factor(df.labelled$label)

#cell#

vec.labelled.valset = rownames(df.labelled[df.labelled$set == 'val',])
vec.labelled.tr_set = rownames(df.labelled[df.labelled$set == 'train',])
vec.labelled.ts_set = rownames(df.labelled[df.labelled$set == 'test',])
############################################################
dffeatures.labelled <- dffeatures[vec.labelled,]
dffeatures.labelled$label  <- df.labelled$label

dffeatures.labelled.devset <- dffeatures.labelled[!(rownames(dffeatures.labelled) %in% vec.labelled.valset),]
dffeatures.labelled.tr_set <- dffeatures.labelled[vec.labelled.tr_set,]
dffeatures.labelled.ts_set <- dffeatures.labelled[vec.labelled.ts_set,]

table(dffeatures.labelled.tr_set$label)


goodrows <- 1 - colSums(sapply(dffeatures.labelled.tr_set, is.na)) / nrow(dffeatures.labelled.tr_set)

names(goodrows[goodrows<0.1])


for (cc in colnames(dffeatures.labelled.tr_set)){
  if (is.factor(dffeatures.labelled.tr_set[,cc]) ){
    setdiff_ = setdiff(dffeatures.labelled.ts_set[,cc], dffeatures.labelled.tr_set[,cc])
    if (length(setdiff_)>0){
      print(cc)
      print(setdiff_)
    }
  }
}


# GLMNET ---------------------------------------------------------------------

library(glmnet)
# Using glmnet to directly perform CV
set.seed(0)

x_train <- model.matrix( ~ .-1, dffeatures.labelled.tr_set[,!(colnames(dffeatures.labelled.tr_set) %in% c("label"))])
dim(x_train)

cvob1=cv.glmnet(x=x_train,
                y=dffeatures.labelled.tr_set[,"label"],
                family="binomial",alpha=1, 
                type.measure="auc", nfolds = 5, lambda = seq(0.001,0.1,by = 0.001),
                standardize=FALSE)
plot(cvob1)

control <- trainControl(method="cv", number=5, returnResamp="all",
                        classProbs=TRUE, summaryFunction=twoClassSummary)
#classProbs = TRUE

tuneGrid <- expand.grid(alpha=c(0.00, 0.25, 0.50, 0.75, 0.99, 1.00), lambda = 10^seq(-5,-2,0.5))
tune = list()
fits = list()
rocs = list()
for (ii in 1:5){
    glmnetFit <- train(label ~ ., data = dffeatures.labelled.tr_set, 
                       method = "glmnet",
                       na.action = na.pass,
                       tuneGrid=tuneGrid,
                       metric = "ROC",
                       trControl = control)
    fits[[ii]] <- glmnetFit
    tune[[ii]] <- glmnetFit$bestTune
    rocs[[ii]] <- max(glmnetFit$results$ROC)
}

tune

varImp(glmnetFit, scale=T)
as.data.frame(glmnetFit$bestTune)

saveRDS(glmnetFit, sprintf("glmnet.rds", Sys.Date()))

## Save predictions  ---------------------------------------------------------

dffeatures[,"predictions_glmnet"] = predict(glmnetFit, newdata = dffeatures, type = "prob", na.action = na.pass)$special

write.table(dffeatures[,c("predictions_glmnet"), drop=F],
            file="all_predictions_glmnet.tab", quote=F, sep='\t')

