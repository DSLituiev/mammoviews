# coding: utf-8
rm(list=ls())

library(caret)
library(gbm3)
library(data.table)
library(ggplot2)
library(fastmatch)

read.gz <- function(filename, ...){
  as.data.frame(fread(paste("zcat < ",filename),
                            header=TRUE,  fill = TRUE, ...))
}

TABLEDIR = "../tables/"
fn_ids = paste(TABLEDIR,
               "2017-06-mammo_tables/df_dcm_reports_birads_path_indic_dens_birad_wi_year_noreport_nodupl.csv.gz", sep='/')

ids = read.gz(fn_ids, select="id")$id

fn_features = paste(TABLEDIR, "mammo_dicom_headers/df_all_mammos_dicom_headers_selected_expanded.tab.gz", sep='/')
dffeatures = read.gz(fn_features, sep='\t')
print(nrow(dffeatures))
print(length(ids))

dffeatures <- dffeatures[fmatch(unique(ids), dffeatures$filename),]
dffeatures <- dffeatures[!is.na(dffeatures$filename),]
rm(ids)

# Data formatting -----------------------------------------

collist = c("BodyPartThickness", "XRayTubeCurrentInuA",  "ContentTime",
            "DetectorTemperature", "WindowCenter", "FieldOfViewRotation")
for (cc in collist){
    dffeatures[,cc] <- as.numeric(dffeatures[,cc])
}


dtypes = sapply(dffeatures, class)
names(dtypes[dtypes == 'character'])


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
  dffeatures[,cc] = as.factor(dffeatures[,cc])
}
#cell#

colSums(sapply(dffeatures, is.na))

# Read labels  --------------------------------


fn.labelledset = paste(TABLEDIR, "spotmag_predictions/train_test_split-2018-02-15-within7e5.csv", sep='/')
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

#cell#

dffeatures.labelled.devset <- dffeatures.labelled[!(rownames(dffeatures.labelled) %in% vec.labelled.valset),]
dffeatures.labelled.tr_set <- dffeatures.labelled[vec.labelled.tr_set,]
dffeatures.labelled.ts_set <- dffeatures.labelled[vec.labelled.ts_set,]

colnames(dffeatures.labelled.tr_set)


for (cc in colnames(dffeatures.labelled.tr_set)){
  if (is.factor(dffeatures.labelled.tr_set[,cc]) ){
    setdiff_ = setdiff(dffeatures.labelled.ts_set[,cc], dffeatures.labelled.tr_set[,cc])
    if (length(setdiff_)>0){
      print(cc)
      print(setdiff_)
    }
  }
}




# GBM3 ----------------------------------------

par_detail <- gbmParallel(num_threads = 4) # Pass to par_details in gbmt
gbmt_fit <- gbmt(label ~ .,
                  data = dffeatures.labelled.tr_set,
                  cv_folds = 10,
                  # training_params = training_params(num_trees = 100, 
                  #                                   interaction_depth = 1,
                  #                                 min_num_obs_in_node = 10, 
                  #                                 shrinkage = 0.005, 
                  #                                 bag_fraction = 0.5,
                  #                                 num_features = 2),
                  keep_gbm_data = TRUE,
                  par_detail=par_detail)

best_iter_cv <- gbmt_performance(gbmt_fit, method='cv')
plot(best_iter_cv)

best.iter.oob <- gbmt_performance(gbmt_fit,method="OOB")  # returns out-of-bag estimated best number of trees
plot(best.iter.oob)

saveRDS(gbmt_fit, sprintf("gbm3_ntrees_%d_%s.rds", best_iter_cv, Sys.Date()))

## Feature Importance Plotting ----------------

infl_gbmt <- (as.data.frame(relative_influence(gbmt_fit, best_iter_cv, rescale=T)))
colnames(infl_gbmt) <- "relative influence"
infl_gbmt[,"variable"] <- rownames(infl_gbmt)

infl_gbmt = infl_gbmt[infl_gbmt$`relative influence` >0,]

plimp <- ggplot(data=infl_gbmt) +
  geom_segment(size=5, colour='blue') + 
  aes(x=reorder(variable,`relative influence`),
      xend = variable,
      y = 2e-6,
      yend=`relative influence`,
      label=`relative influence`) +
  scale_y_log10() + 
  # coord_cartesian(ylim= c(0.8e-6, 1.05)) +
  ylab("relative influence") + xlab("") +
  coord_flip() +
  theme(axis.text.y = element_text(colour="black",size=16,angle=0,face="plain"),
        axis.text.x = element_text(colour="black",size=16,angle=0,face="plain"),
        axis.title.x = element_text(colour="black",size=16,angle=0,face="plain"),
        # panel.background = element_rect(fill = "transparent"), # bg of the panel
        #plot.background = element_rect(fill = "transparent"), # bg of the plot
        # panel.grid.major = element_blank(), # get rid of major grid
         # , panel.grid.minor = element_blank() # get rid of minor grid
          , legend.background = element_rect(fill = "transparent") # get rid of legend bg
          , legend.box.background = element_rect(fill = "transparent") # get rid of legend panel bg
        )

plimp + coord_trans(limy= c(0.5e-6, 1.05)) + coord_flip()
  
plimp + ggsave("img/xgbt_importances.eps", device = 'eps', bg = "transparent",
               width = 8, height = 6, dpi = 300, units = "in" )
plimp + ggsave("img/xgbt_importances.png", device = 'png', bg = "transparent",
               width = 8, height = 6, dpi = 300, units = "in" )


dffeatures[,"predictions_gbmt"] = predict(gbmt_fit, newdata = dffeatures,
                                          n.trees = best_iter_cv,
                                          type = "response", na.action = na.pass)

# GBM-CARET ---------------------------------------------------

control <- trainControl(method = "cv",
                        number = 10, 
                        p =.8, 
                        savePredictions = TRUE, 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)

tuneGrid <- expand.grid(n.trees = c(80,100,120,140,160),
            shrinkage=c(0.025, 0.05, 0.1, 0.2),
            interaction.depth = c(1,2),
            n.minobsinnode = c(10, 15))

gbmFit1 <- train(label ~ .,
                 data = dffeatures.labelled.tr_set, 
                 method = "gbm",
                 na.action = na.pass,
                 tuneGrid=tuneGrid,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 metric = "ROC",
                 trControl = control,
                 # importance = TRUE,
                 verbose = FALSE)
gbmFit1

## Feature Importance Plotting ---------------------------------------------

gbmsmmry <- summary(gbmFit1, normalize=T, plotit=F)

gbmsmmry <- gbmsmmry[gbmsmmry$rel.inf>0,]


ggplot(data=gbmsmmry) +
  geom_segment(size=3, colour='red') + 
  aes(x=reorder(var,rel.inf, sum),
      xend = var,
      y = 0.002,
      yend=(rel.inf),
      label=rel.inf) +
  scale_y_log10() + 
  ylab("relative influence") + xlab("") +
  coord_flip()

saveRDS(gbmFit1, "gbm_ntrees80_interactiondepth2_shrinkage0.2_nminobsinnode15_trainset_2018-02-18.rds")

dffeatures[,"predictions_gbm"] = predict(gbmFit1, newdata = dffeatures, type = "prob", na.action = na.pass)$special

# RPART -----------------------------------------------------------------

tuneGrid <- expand.grid(cp=c(0.0, 0.0125, 0.025, 0.05, 0.1, 0.2))

rpartFit1 <- train(label ~ ., data = dffeatures.labelled.tr_set, 
                   method = "rpart",
                   na.action = na.pass,
                   tuneGrid=tuneGrid,
                   ## This last option is actually one
                   ## for gbm() that passes through
                   metric = "ROC",
                   trControl = control
)
varImp(rpartFit1)


predictions.ts_set = predict(rpartFit1, 
                             newdata = dffeatures.labelled.ts_set,
                             type='prob', na.action = na.pass)

dffeatures[,"predictions_rpart"] = predict(rpartFit1, newdata = dffeatures, type = "prob", na.action = na.pass)$special

# XGB ---------------------------------------------------------------------
control <- trainControl(method="cv", number=10)
#classProbs = TRUE

#tuneGrid <- expand.grid(cp=c(0.0, 0.0125, 0.025, 0.05, 0.1, 0.2))
xgbFit <- train(label ~ ., data = dffeatures.labelled.tr_set, 
                   method = "xgbTree",
                   na.action = na.pass,
                   #tuneGrid=tuneGrid,
                   metric = "Accuracy",
                   trControl = control)

varImp(xgbFit, scale=T)

as.data.frame(xgbFit$finalModel$params)

xgbFit$bestTune

saveRDS(xgbFit, sprintf("xgbtree_maxdepth1_subsample1_eta0.3_%s.rds", Sys.Date()))

predictions.ts_set = predict(xgbFit, 
                             newdata = dffeatures.labelled.ts_set,
                             type='prob', na.action = na.pass)


## Save all predictions  ---------------------------------------------------------

dffeatures[,"predictions_xgb"] = predict(xgbFit, newdata = dffeatures, type = "prob", na.action = na.pass)$special

write.table(dffeatures[, c(grep('prediction',colnames(dffeatures), value=T),
                           "ViewModifierCodeMeaning", "ViewCodeValue")],
            file = "all_predictions_allmodels_trained_on_train.tab", quote=F, sep='\t')

