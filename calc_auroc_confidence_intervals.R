rm(list=ls())
library(pROC)
library(ggplot2)
library(ggsignif)
library(dplyr)
library(data.table)
read.gz <- function(filename, ...){
  as.data.frame(fread(paste("zcat < ",filename),
                      header=TRUE,  fill = TRUE, ...))
}


tag <- "e5ce2d69b035975cb5336cec0da9a32a"
fnall <- "../tables/all_predictions_with_images.tab"
fnall <- paste0("../tables/all_predictions_with_images-", tag,".tab")

predictions <- as.data.frame(fread(fnall, sep='\t'), header=TRUE,  fill = TRUE)

labelled <- sapply(predictions$label, function(x) nchar(x)>0)

print(nrow(predictions[labelled,]))
predictions <- predictions[labelled,]


predictions[,'ViewModifier'] <- as.numeric(predictions[,'ViewModifier']!='')

predictions[, "label"] <- factor(predictions[, "label"], c('normal', 'special'))

predictions[,"view"] <- factor(predictions[,"view"], c('N','M','T','W','X'))
head(predictions)
# holdout <- predictions[predictions$set == 'val',]

ggplot(holdout, aes(view, `score_max_wire_image+gbmt`)) + geom_point()

validation <- predictions[predictions$set == 'test',]

clmns <- colnames(predictions)

othercols <- c('id', 'set', 'view', 'label')
modelnames <- c('ViewModifier', 'rpart', 'gbm', 'glmnet','xgb', 'gbmt',
                'image',
                'image_max',
                'wire',
                'wire_max',
                'max_image_wire_max',
                'image+gbmt',
                'max_wire_max_image+gbmt',
                'max_image_wire',
                'max_wire_image+gbmt')



clean_score_names <- function(x){
  return( gsub('score_', '', x) )
  # paste(strsplit(x, '_')[[1]][-1],collapse='_')
}

clmns_clean <-  vapply(clmns, clean_score_names, '')

cols_ <-  factor(vapply(colnames(predictions) , clean_score_names, ''),
                 c(othercols,modelnames))

colnames(validation) <-  cols_

validation <- validation[,!is.na(colnames(validation))]

cols_ <- cols_[!is.na(cols_)]
cols_ <- cols_[order(cols_)]

validation <- validation[,as.character(cols_)]

colnames(validation)
# clmns <-clmns[vapply(clmns, function(x) strsplit(x, '_')[[1]][1]=='score', TRUE)]

## Perform McNemars test for prediction difference ----------------------------------------------------

mcnemar.test(table(validation$`max_wire_max_image+gbmt`>0.5, validation$max_image_wire_max>0.5))

mcnemar.test(table(validation$`max_wire_max_image+gbmt`>0.5, validation$gbmt>0.5))

## Calculate significance of pairwise auROC differences -----------------------------------------------
cis <- list()
rocobjects <- list()
ii <- 0
for (clmn in modelnames){
  # ii = 1
  print('====================')
  print(clmn)
  rocobj   <- plot.roc(  validation[, "label"],
                         validation[,clmn],
                         levels = (levels(validation[, "label"])),
                         xlim = c(100,0),
                         ylim = c(0,100),
                         percent=TRUE,
                         print.auc=TRUE)
  rocobjects[[clmn]] <- rocobj
  cis[[clmn]] <- ci(rocobj, of="auc", thresholds="best")
}

## Wire model on wire cases
for (clmn in c('wire', 'wire_max')){
  print('====================')
  print(clmn)
  rocobj   <- plot.roc(  validation[, "view"]=='W',
                         validation[,clmn],
                         # levels = (levels(validation[, "label"])),
                         xlim = c(100,0),
                         ylim = c(0,100),
                         percent=TRUE,
                         print.auc=TRUE)
  rocobjects[[clmn]] <- rocobj
  cis[[paste0(clmn, ' (vs other views)')]] <- ci(rocobj, of="auc", thresholds="best")
}
###
modelnames <- c('ViewModifier', 'rpart', 'gbm', 'glmnet','xgb', 'gbmt',
                'image', "image_max",
                'wire', 'wire_max',
                'wire (vs other views)', 'wire_max (vs other views)',
                'max_image_wire_max',
                'image+gbmt',
                'max_wire_max_image+gbmt')

##

dfcis <- as.data.frame(t(do.call(cbind.data.frame, lapply(cis, as.vector))))
colnames(dfcis) <- c('lower', 'auROC', 'upper')

dfcis[,"model"] <- factor(rownames(dfcis),
                           modelnames)

dfcis <-  dfcis[!is.na(dfcis[,"model"]),]

rownames(dfcis) <- dfcis[,"model"] 

dfcis <- dfcis[modelnames,]


# dfcis <-dfcis %>% mutate(model = factor(model, levels=rev(levels(model))))
dfcis_nowire <- dfcis[!(rownames(dfcis) %in% c('wire','wire_max')),]
dfcis_nowire$model <-  factor(dfcis_nowire$model)
# 
# 
# annotation_df <- data.frame(color=c("E", "H"), 
#                             start=c("Good", "Fair"), 
#                             end=c("Very Good", "Good"),
#                             y=c(3.6, 4.7),
#                             label=c("Comp. 1", "Comp. 2"))

roc.test(rocobjects[["ViewModifier"]], rocobjects[["gbmt"]])

## Format Pairwise comparisons

keys <- names(rocobjects)
dfcompar <- data.frame()
for (a in 1:length(rocobjects)){
  for (b in 1:a){
    na <- keys[a]
    nb <- keys[b]
    if ((as.numeric(rocobjects[[na]]$auc)==100)||(as.numeric(rocobjects[[nb]]$auc)==100)){
      dfcompar[na, nb] <- NA
    } else {
      dfcompar[na, nb] <- roc.test(rocobjects[[na]], rocobjects[[nb]], method='delong')$p.value
    }
  }
}


fn.comparison <- paste0("../tables/auroc_delong_comparison-", tag,".csv")
write.csv(dfcompar, file=fn.comparison)
