
install.packages("stringr")
install.packages("car")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("xlsx")
install.packages("reshape2")
install.packages("reshape")
install.packages('ISLR')
install.packages("readr")
install.packages('caret')
install.packages('caTools')
install.packages('rpart.plot')
install.packages('rpart')
install.packages("party")
install.packages("partykit")
install.packages("package:stats")
install.packages('vcd')
install.packages('mltools')

library(dplyr)
library(ggplot2)
library(MASS) 
library(reshape2) 
library(reshape) 
library(corrplot)

library(scales)
library(plyr)
library(caTools)
library(rpart)
library(rpart.plot)
library(tidyverse) 
library(caret) 
library(ISLR)
library(readr)
library(party)
library(partykit)
library(stringr)
library(car)
require(xgboost)
require(Matrix)
require(data.table)

#Part 1: Classification
churn_Data <- read.csv("C:/Users/abood/Downloads/Assignment 2/Churn Dataset.csv")
head(churn_Data)


str(churn_Data)
summary(churn_Data)
#***********************************************************************
#i.Generate a scatterplot matrix to show the relationships between the variables and a correlation matrix to determine correlated attributes            

  
DF_Var <- subset(churn_Data, select = c("tenure","MonthlyCharges","TotalCharges"))
pairs(DF_Var)

#use_condaenv("r-tensorflow")

install.packages("corrplot")
source("http://www.sthda.com/upload/rquery_cormat.r") # to use rquery.cormat
rquery.cormat(DF_Var)
cormat<-rquery.cormat(DF_Var, graphType="heatmap")

#calculate correlation matrix
correlationMatrix <- cor(DF_Var)
print(correlationMatrix[,1:3])
# Correlation matrix
corrplot(cor(DF_Var),        # Correlation matrix
         method = "circle",                # Correlation plot method (method = number, circle, pie, or color)
         type = "full",                   # Correlation plot style (also "upper" and "lower")
         diag = TRUE,                     # If TRUE (default), adds the diagonal
         tl.col = "black",                # Labels color
         bg = "white",                    # Background color
         title = "",                      # Main title
         col = NULL,                      # Color palette
         tl.cex =0.7,
         cl.ratio =0.2)                            


P = cor(DF_Var)
corrplot(P, method = 'color') 
corrplot(P, method="circle")


#***********************************************************************

#ii.Ensure data is in the correct format for downstream processes and address missing data

churndata<- na.omit(churn_Data)

churndata <- data.frame(churndata)

churndata$gender <- ifelse(churndata$gender == "Male",1,0)
churndata$Partner <- ifelse(churndata$Partner == "Yes",1,0)
churndata$Dependents <- ifelse(churndata$Dependents == "Yes",1,0)
churndata$PhoneService <- ifelse(churndata$PhoneService == "Yes",1,0)
churndata$MultipleLines <- ifelse(churndata$MultipleLines == "Yes",1,ifelse(churndata$MultipleLines == "No",0, -1))
churndata$InternetService <- ifelse(churndata$InternetService == "No",0,ifelse(churndata$InternetService == "DSL",1, 2))
churndata$OnlineSecurity <- ifelse(churndata$OnlineSecurity == "No",0,ifelse(churndata$OnlineSecurity == "Yes",1, -1))
churndata$OnlineBackup <- ifelse(churndata$OnlineBackup == "No",0,ifelse(churndata$OnlineBackup == "Yes",1, -1))
churndata$DeviceProtection <- ifelse(churndata$DeviceProtection == "No",0,ifelse(churndata$DeviceProtection == "Yes",1, -1))
churndata$TechSupport <- ifelse(churndata$TechSupport == "No",0,ifelse(churndata$TechSupport == "Yes",1, -1))
churndata$StreamingTV <- ifelse(churndata$StreamingTV == "No",0,ifelse(churndata$StreamingTV == "Yes",1, -1))
churndata$StreamingMovies <- ifelse(churndata$StreamingMovies == "No",0,ifelse(churndata$StreamingMovies == "Yes",1, -1))
churndata$PaperlessBilling <- ifelse(churndata$PaperlessBilling == "No",0,ifelse(churndata$PaperlessBilling == "Yes",1, -1))
churndata$Churn<- ifelse(churndata$Churn== "No",0,ifelse(churndata$Churn== "Yes",1, -1))
churndata$Contract <- ifelse(churndata$Contract == "Month-to-month",0,ifelse(churndata$Contract == "One year",1, 2))
churndata$PaymentMethod <- ifelse(churndata$PaymentMethod == "Bank transfer (automatic)" | churndata$PaymentMethod == "Credit card (automatic)",2,
                                  ifelse(churndata$PaymentMethod == "Electronic check",1, 0))

DF_Churn <- subset(churndata , select = - customerID)


#Checking the outliers 
boxplot(DF_Churn$tenure,
        main = 'boxplot for tenure column ', 
        ylab = 'tenure',
        horizontal = TRUE )


boxplot(DF_Churn$MonthlyCharges,
        main = 'boxplot for MonthlyCharges column ', 
        ylab = 'MonthlyCharges',
        horizontal = TRUE )

boxplot(DF_Churn$TotalCharges,
        main = 'boxplot for TotalCharges column ', 
        ylab = 'TotalCharges',
        horizontal = TRUE )




DF_Var <- subset(DF_Churn, select = -Churn)
churn_DF <- sapply(DF_Churn,unclass)



#***********************************************************************

#iii.Split the dataset into 80 training/20 test set and fit a decision tree to the training data. Plot the tree, and interpret the results.


install.packages("caret")
install.packages("numDeriv")
library(caret)
set.seed(123)
split_train_test <- createDataPartition(DF_Churn$Churn,p=0.8,list=FALSE)
dtrain<- DF_Churn[split_train_test,]
dtest<-  DF_Churn[-split_train_test,]


library(rpart.plot)

#decision tree
tree_fit <- rpart(Churn ~., data = dtrain, method="class")

rpart.plot(tree_fit)
library(party)

#plot conditional parting plot
ctree_ <- ctree(Churn ~ ., dtrain)
plot(ctree_)



#***********************************************************************
#. Try different ways to improve your decision tree algorithm
set.seed(123) 

#specify the cross-validation method
CT <- trainControl(method = "cv", number = 10)

#Fit a decision tree model and use k-fold CV to evaluate performance
Dtree_fit <- train(Churn ~ . , data = dtrain,
                   method = "rpart",
                   parms = list(split = "gini"), 
                   trControl = CT,
                   tuneLength = 100)

#Step 5: Evaluate - view summary of k-fold CV               
print(Dtree_fit)  

#View final model
Dtree_fit$finalModel
prp(Dtree_fit$finalModel, box.palette = "Reds", tweak = 1.2)
Dtree_fit$resample

#Check accuracy

Dtree_fit[(Dtree_fit>=0.2)] = 1
Dtree_fit[(Dtree_fit<0.2)] = 0
pred1 = predict(Dtree_fit,newdata = dtest , type="class")
confusionMatrix(as.factor(dtest$Churn),pred1 )  
#information gain
Dtree_fit_info <- rpart(Churn ~ . , data = dtrain,
                        method = "class",
                        parms = list(split = "information"))
pred2=predict(Dtree_fit_info,newdata = dtest , type="class")
confusionMatrix(as.factor(dtest$Churn),pred2 )  


Hr_Model <- rpart(Churn ~ ., data = dtrain, method = "class", 
                  control = rpart.control(cp = 0))
summary(Hr_Model)
rpart.plot(Hr_Model)
# Compute the accuracy of the pruned tree
dtest$pred <- predict(Hr_Model, dtest, type = "class")
base_accuracy <- mean(dtest$pred == dtest$Churn)
confusionMatrix(as.factor(dtest$Churn),dtest$pred)
#prepurning
# Grow a tree with minsplit of 100 and max depth of 8
hr_model_preprun <- rpart(Churn ~ ., data = dtrain, method = "class", 
                          control = rpart.control(cp = 0, maxdepth = 8,minsplit = 100))
# Compute the accuracy of the pruned tree
dtest$pred <- predict(hr_model_preprun, dtest, type = "class")
accuracy_preprun <- mean(dtest$pred == dtest$Churn)


#Postpruning
# Prune the Hr_Model based on the optimal cp value
hr_model_pruned <- prune(Hr_Model, cp = 0.0048 )
# Compute the accuracy of the pruned tree
dtest$pred <- predict(hr_model_pruned, dtest, type = "class")
accuracy_postprun <- mean(dtest$pred == dtest$Churn)
data.frame(base_accuracy, accuracy_preprun, accuracy_postprun)
#***********************************************************************
# vi.5)Classify the data using the XGBoost model with nrounds = 70 and max depth = 3. Evaluate the performance

# get the numb 80/20 training test split
X_train = data.matrix(dtrain[,-20])  # independent variables for train

y_train = dtrain[,20]                                # dependent variables for train

X_test = data.matrix(dtest[,-20])                    # independent variables for test

y_test = dtest[,20]                                   # dependent variables for test

# put our testing & training data into two seperates Dmatrixs objects
install.packages("xgboost")
library(xgboost)
train_set = xgb.DMatrix(data = X_train, label= y_train)
test_set = xgb.DMatrix(data = X_test, label= y_test)

# train a model using our training data
model <- xgboost(data = train_set, # the data   
                 nround = 70, # max number of boosting iterations
                 max.depth=3,
                 objective = "binary:logistic")  # the objective function

# train a model using our training data
                            # max number of boosting iterations
summary(model)


############## accuracy of the pruned tree #############
y_pred_xgboost <- predict(model, newdata= test_set)
y_pred_xgboost

y_pred_xgboost[(y_pred_xgboost>=.5)] = 1
y_pred_xgboost[(y_pred_xgboost<.5)] = 0
confusionMatrix(factor(y_pred_xgboost, levels = c(0,1)), factor(y_test, levels = c(0,1)) 
                , mode= 'everything')

###################################################
#use model to make predictions on train data

pred_train =  predict(model , newdata =as.matrix(X_train) )

pred_train = ifelse(pred_train>.5 , 1 , 0)
install.packages("caret")
library(caret)
confusionMatrix(factor(pred_train, levels = c(0,1)), factor(y_train, levels = c(0,1)) 
                , mode= 'everything')




y_pred_xgboost_nm = as.numeric(y_pred_xgboost)
y_test_nm = as.numeric(y_test)
class(y_test_nm)


install.packages("ROSE")
library(ROSE) 
# ROC and AUC
ROSE::roc.curve(y_test_nm, y_pred_xgboost_nm) #0.825



#vii. Train a deep neural network using Keras with 3 dense layers.
library(keras)
use_condaenv("tf")
library(reticulate)
install_keras(method = c("conda"), conda = "auto", version = "default", tensorflow = "gpu")
tensorflow::tf_config()
tensorflow::install_tensorflow()
# Creating the model
install.packages("keras")
install.packages("tensorflow")
install.packages("reticulate")


library(tensorflow)
library(tidyverse)

model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'tanh', input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'tanh') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)
# Compiling the model
model %>% compile(loss = "categorical_crossentropy",
                  optimizer = "adam",
                  metrics = c("accuracy"))
history <- model %>% 
  fit(x_train,
      y_train,
      epoch = 10,
      batch_size = 256,
      validation_split = 0.1,
      verbose = 2)
plot(history)
model %>% 
  evaluate(x_test,
           y_test)
#7.Compare the performance of the models in terms of the following criteria: precision, recall, accuracy, Fmeasure. Identify the model that performed best and worst according to each criterion.


#for DT
confusionMatrix(factor(dtest$pred, levels = c(0,1)), factor(y_test, levels = c(0,1)) 
                , mode= 'everything')
#for xgboost
confusionMatrix(factor(y_pred_xgboost, levels = c(0,1)), factor(y_test, levels = c(0,1)) 
                , mode= 'everything')
#for NN
confusionMatrix(factor(y_pred_NN, levels = c(0,1)), factor(y_test, levels = c(0,1)) 
                , mode= 'everything')
#viii. Carry out a ROC analysis to compare the performance of the DT, XGboost & NN techniques. Plot the ROC graph of the models
install.packages('pROC')
library(pROC)
install.packages("plotROC")
library(ggplot2)
library(plotROC)

#1. Neural Networks
Test = test_set
PPR = y_pred_NN
ROC_score=roc(Test, PPR) 
print(ROC_score)
plot(ROC_score ,main ="ROC curve -- Neural Networks ")

#2. Decision Tree
PRE =dtest$pred
ACT = y_test
ROC_Score=roc( PRE, ACT ) 
print(ROC_Score)
plot(ROC_Score ,main ="ROC curve -- Decision Tree ")

#3. XGBoost Model 
TE = y_test
PR = y_pred_xgboost
ROC_Score=roc(TE, PR) #AUC score
print(ROC_Score)
plot(ROC_Score ,main ="ROC curve -- XGBoost Model  ")








