### Practical Machine Learning Course Project
=============================================

#### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. As requested, the goal of this short project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, who were asked to perform barbell lifts correctly and incorrectly in 5 different ways, and **predict the manner in which they did the exercise, using any of the other variables to predict with**. More information about the data is available from http://groupware.les.inf.puc-rio.br/har (section on the Weight Lifting Exercise Dataset).

Within these premises, this report describes the random forest model finally built, how cross validation was used, which choices were done for the cleaning of data, the exploration of data and the preprocessing, after a few trade-offs of the different alternatives, resulting in an out of sample error below 0,5%, confirming the suitability of the model designed. Finally, 1-time validation was done to predict the 20 different test cases provided, which resulted 100% accurate.

#### Initialization
Loading of library, suppressing results and messages:

```r
setInternet2(use = TRUE)
library(caret)
library(randomForest)
```


#### Getting and cleaning the data 

- Download data (original data files expected in default working directory):

```r
if (!file.exists("pml-training.csv")) {
    URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download.file(URL, destfile = "pml-training.csv")
}
if (!file.exists("pml-testing.csv")) {
    URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(URL, destfile = "pml-testing.csv")
}
```


- Read data:

```r
data <- read.csv("pml-training.csv", header = TRUE, sep = ",")
tstc <- read.csv("pml-testing.csv", header = TRUE, sep = ",")
```

Total data frame has 19622 rows (lifts recorded) and 160 columns, i.e. 159 predictors.

#### Cleaning the data, Exploratory Data Analysis and Preprocessing

- Cleaning the data: first columns seems not to correspond to sensor data, but to username, timestamps or window, so they were removed as they are not expected to add much value to the prediction model:

```r
# summary(data) str(data)
sum(complete.cases(data))
```

```
## [1] 406
```

```r
data <- data[, -c(1:7)]
tstc <- tstc[, -c(1:7)]
```

A lot of "zeroed" columns and "NAs" were detected, so such 59 "unique value" predictors were removed using nearZeroVar function (to remove zero covariates):

```r
nearZero <- nearZeroVar(data[, -153])
data <- data[, -nearZero]
tstc <- tstc[, -nearZero]
```

Moreover, those 41 columns with a majority of "NAs" (19,216 out of 19,622 records) were removed as well:

```r
na_count <- sapply(data, function(x) {
    sum(is.na(x))
})
table(na_count)
```

```
## na_count
##     0 19216 
##    53    41
```

```r
colnotNA <- colSums(is.na(data)) == 0
data <- data[, colnotNA]
tstc <- tstc[, colnotNA]
dim(data)
```

```
## [1] 19622    53
```

```r
sum(complete.cases(data))
```

```
## [1] 19622
```

Number of predictors goes then down to 52, while the number of complete cases grows from 406 to all 19,622 lifts recorded.

- More Exploratory Data Analysis: outcome variable is a factor variable quite evenly distributed among its 5 values. 

```r
table(data$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

Predictors are a mix of integer, numeric and factor classes, which has to be taken into account in any preprocessing.The only factor remaining after the cleaning of data performed is the outcome variable.

```r
table(sapply(data, class))
```

```
## 
##  factor integer numeric 
##       1      25      27
```

- Preprocessing: after the corresponding trade-offs, I decided to abort any regularization of the diverse numeric predictors, not so critical for the intended random forest model, or preprocessing using Principal Component Analysis, which did not add much value either in any of the trials performed.

```r
preProc <- preProcess(data[, -53], method = "pca", thresh = 0.98)
preProc
```

```
## 
## Call:
## preProcess.default(x = data[, -53], method = "pca", thresh = 0.98)
## 
## Created from 19622 samples and 52 variables
## Pre-processing: principal component signal extraction, scaled, centered 
## 
## PCA needed 31 components to capture 98 percent of the variance
```

In comparison with the results above for 98% of the variance, for 95% of 25 components were needed by PCA, and for 99% 36 components.

Same comments above apply to correlation analysis: a few predictors show a high correlation, but a more comprehensive analysis would be required, and only in case of severe needs to enhance performance, which did not apply.

```r
M <- abs(cor(data[, -53]))
diag(M) <- 0
M[which(M > 0.9, arr.ind = TRUE)]
```

```
##  [1] 0.9809 0.9249 0.9920 0.9657 0.9809 0.9278 0.9749 0.9657 0.9249 0.9278
## [11] 0.9334 0.9920 0.9749 0.9334 0.9182 0.9182 0.9790 0.9145 0.9790 0.9330
## [21] 0.9145 0.9330
```

```r
unique(rownames(M[which(M > 0.9, arr.ind = TRUE), ]))
```

```
##  [1] "total_accel_belt" "accel_belt_y"     "accel_belt_z"    
##  [4] "accel_belt_x"     "roll_belt"        "pitch_belt"      
##  [7] "gyros_arm_y"      "gyros_arm_x"      "gyros_dumbbell_z"
## [10] "gyros_forearm_z"  "gyros_dumbbell_x"
```

#### Split of data for Cross Validation

80% of data was allocated to the training set and the other 20% for testing set to estimate model accuracy and out of sample model, while leaving the test cases provided for a 1-time validation of the prediction model design.Seed is set to enable reproducible research.

```r
set.seed(12)
trainIndex = createDataPartition(data$classe, p = 0.8, list = FALSE)
training = data[trainIndex, ]
testing = data[-trainIndex, ]
dim(training)
```

```
## [1] 15699    53
```

```r
dim(testing)
```

```
## [1] 3923   53
```


#### Prediction Model

Generalized linear model was discarded, as it can only be used for 2-class outcome, and here we have a 5-class outcome. Linear regression does not seem a priori too applicable either. After analyzing rpart method and its variability of results, *random forest was selected*, being a priori an efficient-enough solution for this classification problems where accuracy rules, and also reduces the need of strict regularization and preprocessing. Parameter ntree was finally set to 1024, instead of the default 512.

```r
set.seed(12)
modelFit <- randomForest(classe ~ ., data = training, ntree = 1024)
modelFit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, ntree = 1024) 
##                Type of random forest: classification
##                      Number of trees: 1024
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.41%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4460    3    0    0    1   0.0008961
## B   11 3024    3    0    0   0.0046083
## C    0   14 2722    2    0   0.0058437
## D    0    0   23 2548    2   0.0097163
## E    0    0    1    5 2880   0.0020790
```

**In sample error is of 0.41%, i.e. an accuracy of 99.59% on the training data set, which seems acceptable**. Being below 100%, the risk of overfitting seems under control.

#### Confusion matrix and Out of Sample Error

Applying the prediction function created to the testing data set, as splitted above, the following confusion matrix is obtained:

```r
confMat <- confusionMatrix(testing$classe, predict(modelFit, testing))
confMat
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  759    0    0    0
##          C    0    0  684    0    0
##          D    0    0    1  642    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    0.999    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    0.998    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.175    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    0.999    1.000    1.000
```

An accuracy of 99.97% is obtained, which can be considered as an outstanding (and unexpected) result. **Out of Sample Error is therefore of 0.03%**.These results validate the prediction model and the hypothesis and decisions performed, so no further refinements seem to be required, like traincontrol, bagging, or boosting.

The 10 more important predictors are ranked below:

```r
importance <- varImp(modelFit)
importance$Variable <- row.names(importance)
head(importance[order(importance$Overall, decreasing = T), ], 10)
```

```
##                   Overall          Variable
## roll_belt          1025.0         roll_belt
## yaw_belt            729.7          yaw_belt
## pitch_forearm       634.2     pitch_forearm
## magnet_dumbbell_z   615.9 magnet_dumbbell_z
## pitch_belt          564.1        pitch_belt
## magnet_dumbbell_y   534.5 magnet_dumbbell_y
## roll_forearm        495.1      roll_forearm
## magnet_dumbbell_x   389.7 magnet_dumbbell_x
## roll_dumbbell       332.7     roll_dumbbell
## accel_dumbbell_y    320.9  accel_dumbbell_y
```


#### Validation

Finally, the prediction model is applied to the validation data set, as provided by the course staff, which resulted in a 100% accuracy.

```r
predict(modelFit, tstc)
```

