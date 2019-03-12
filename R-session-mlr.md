R Session - mlr Package
=======================

This notebook shows how to use the `mlr` package to train two classifiers and evaluate them. You can run the code by yourself by loading this file from RStudio.

Installing and loading the package
----------------------------------

`mlr` is available at CRAN, so you can install as any other R package through the `install.packages()` function. Development version is available in GitHub.

~~~~ {.r}
# install.packages("mlr")                   # Uncomment this line if you need to install the package from CRAN
# remotes::install_github("mlr-org/mlr")    # Uncomment this line if you want to install the latest version (in development) from GitHub

library(mlr)
~~~~

    ## Loading required package: ParamHelpers

Prepare the data
----------------

~~~~ {.r}
# To work with data in an mldr object we need to apply some  changes
mld <- mldr.datasets::flags
mldmlr <- data.frame( # Builds a data.frame changing the type of labels to logical
                 mld$dataset[,mld$attributesIndexes], 
                 sapply(mld$dataset[,mld$labels$index], as.logical))
                 
mldmlr[sapply(mldmlr, is.character)] <-  # Change "character" columns to "factor"
            lapply(mldmlr[sapply(mldmlr, is.character)], as.factor)
~~~~

Create the mlr task
-------------------

~~~~ {.r}
mldtask <- makeMultilabelTask(id = "MLLTest", 
                  data = mldmlr, target = row.names(mld$labels))

mldtask # Explore the mlr task object
~~~~

    ## Supervised task: MLLTest
    ## Type: multilabel
    ## Target: red,green,blue,yellow,white,black,orange
    ## Observations: 194
    ## Features:
    ##    numerics     factors     ordered functionals 
    ##          10           9           0           0 
    ## Missings: FALSE
    ## Has weights: FALSE
    ## Has blocking: FALSE
    ## Has coordinates: FALSE
    ## Classes: 7
    ##    red  green   blue yellow  white  black orange 
    ##    153     91     99     91    146     52     26

Train two classifiers
---------------------

~~~~ {.r}
# Test one adaptation-based algorithm and one transformation-based
class1 <- makeLearner("multilabel.randomForestSRC")
class2 <- makeMultilabelBinaryRelevanceWrapper(
              makeLearner("classif.rpart", predict.type = "prob"))
model1 <- train(class1, mldtask, subset = 1:150)
model2 <- train(class2, mldtask, subset = 1:150)
~~~~

Test and evaluate the classifiers
---------------------------------

~~~~ {.r}
predict1 <- predict(model1, task = mldtask, subset = 151:194)
predict2 <- predict(model2, task = mldtask, subset = 151:194)

measures <- list(multilabel.hamloss, multilabel.f1, multilabel.acc, multilabel.subset01)

performance(predict1, measures = measures)
~~~~

    ##  multilabel.hamloss       multilabel.f1      multilabel.acc 
    ##           0.2305195           0.7517920           0.6570346 
    ## multilabel.subset01 
    ##           0.7272727

~~~~ {.r}
performance(predict2, measures = measures)
~~~~

    ##  multilabel.hamloss       multilabel.f1      multilabel.acc 
    ##           0.3051948           0.6511424           0.5324134 
    ## multilabel.subset01 
    ##           0.9090909
