R Session - utiml Package
=========================

This notebook shows how to use the `utiml` package to train and evaluate a couple of multi-label datasets. You can run the code by yourself by loading this file from RStudio.

Installing and loading the package
----------------------------------

`utiml` is available at CRAN, so you can install as any other R package through the `install.packages()` function. Development version is available in GitHub.

``` r
# install.packages("utiml")                   # Uncomment this line if you need to install the package from CRAN
# remotes::install_github(""rivolli/utiml"")    # Uncomment this line if you want to install the latest version (in development) from GitHub

library(utiml)
```

    ## Loading required package: mldr

    ## Loading required package: parallel

    ## Loading required package: ROCR

    ## Loading required package: gplots

    ## 
    ## Attaching package: 'gplots'

    ## The following object is masked from 'package:stats':
    ## 
    ##     lowess

Prepare the data
----------------

``` r
partitions <- create_holdout_partition(mldr::emotions, c(train = 0.7, test = 0.3))
```

Train two classifiers
---------------------

``` r
# install.packages("e1071")  # Uncomment this line to install the needed package to run SVM as base classifier for ECC

# Test one adaptation-based algorithm and one transformation-based
class1 <- ecc(partitions$train)
class2 <- mlknn(partitions$train)
```

Test and evaluate the classifiers
---------------------------------

``` r
predict1 <- predict(class1, partitions$test)
predict2 <- predict(class2, partitions$test)

measures <- c("hamming-loss", "F1", "accuracy")

multilabel_evaluate(partitions$test, predict1, measures)
```

    ##     accuracy           F1 hamming-loss 
    ##    0.5838015    0.6672285    0.1938202

``` r
multilabel_evaluate(partitions$test, predict2, measures)
```

    ##     accuracy           F1 hamming-loss 
    ##    0.4030899    0.4717228    0.2631086
