
library(caret)

# Let's use all cores in the system.
library(doMC)
registerDoMC(cores=8)

# ROC curve
library(ROCR)

training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

remove_columns <- function(df, cols_to_remove) {
  return(df[, -which(names(df) %in% cols_to_remove)])
}

# Delete time related columns as correct activity doesn't rely on time.
time_columns <- c("raw_timestamp_part_1", 
                  "raw_timestamp_part_2", 
                  "cvtd_timestamp",	
                  "new_window",	
                  "num_window")
training <- remove_columns(training, time_columns)
testing <- remove_columns(testing, time_columns)

# Delete columns that doesn't contain interesting data.
zero_columns <- nearZeroVar(training)
training <- training[, -zero_columns]
testing <- testing[, -zero_columns]

# There are columns with NA. And when there is NA, there are too many. So, imputation won't help.
# Delete columns with NAs.
na_columns <- names(training)[which(
  sapply(names(training), function(col_name) { sum(is.na(training[, col_name])) != 0 }))]
training <- remove_columns(training, na_columns)
testing <- remove_columns(testing, na_columns)

# Delete row number and user name as they're IDs. If we include them, models will use them
# too heavily and won't look at real features.
id_columns <- c("X", "user_name")
training <- remove_columns(training, id_columns)
testing <- remove_columns(testing, id_columns)

# Make sure that column names match.
ncol(training) == ncol(testing)
num_cols <- ncol(training)
all.equal(names(training)[1:(num_cols-1)], names(testing)[1:(num_cols-1)])

# Prepare the validation data. Let's make it's size the same with testing (which is 20).
in_train <- createDataPartition(training$classe, p=(NROW(training) - NROW(testing))/NROW(training), list=FALSE)
validation <- training[-in_train, ]
training <- training[in_train, ]

# Run randomForest as it's usually good model.

(rf_base <- train(classe ~ ., data=training, method="rf"))  # default tree size 500
save(rf_base, file="rf_base.RData")


# There is bias in class distribution!
table(training$classe)

# See if this is affecting the model performance. And the answer is "yes it is" as class.error
# is higher for B, C, D and E than A.
rf_base$finalModel

# Tell random forest to sample by strata.
# Reference: http://www.r-bloggers.com/down-sampling-using-random-forests/
(rf_strata <- train(classe ~ ., data=training, method="rf", 
                    strata=training$classe, sampsize=rep(min(table(training$classe)), 5)))

# Did strata fix the issue?
rf_strata$finalModel
save(rf_strata, file="rf_strata.RData")


# PCA then RF
(rf_strata_pca <- train(classe ~ ., data=training, method="rf", preProcess="pca",
                        strata=training$classe, sampsize=rep(min(table(training$classe)), 5)))
save(rf_strata_pca, file="rf_strata_pca.RData")

# Report ROC for validation!
verify <- function(model) {
  confusionMatrix(predict(model, validation), validation$classe)
}

verify(rf_base)
verify(rf_strata)
verify(rf_strata_pca)

# They look all good. Let's use rf_strata as it has improved accuracy for C, D, and E.
pml_write_files <- function(x){
  n <- length(x)
  for(i in 1:n){
    filename <- paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predict(rf_strata, testing))