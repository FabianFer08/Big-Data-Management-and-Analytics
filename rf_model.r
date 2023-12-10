install.packages("dplyr")                       # Install dplyr for data manipulation
library("dplyr")                                # Load dplyr

# Installing the package 
install.packages("caTools")    # For Logistic regression 
library(caTools)

install.packages('randomForest') # For generating random forest model
library(randomForest)

install.packages('caret')                    # classification and regression training : The library caret has a function to make prediction.
library(caret)
install.packages('e1071', dependencies=TRUE)

library(tidymodels)
library(tidyverse)
library(skimr)

analysis_data <- read.csv("data/dt_realestate_train_2022.csv", header = TRUE, sep= ',') 

prediction_data <- read.csv("data/dt_realestate_test_2022.csv", header = TRUE, sep= ',')

dim(analysis_data)
summary(analysis_data)

real_estate_split <- initial_split(analysis_data, prop = 0.7)

real_estate_train <- training(real_estate_split)
real_estate_test <- testing(real_estate_split)

dim(real_estate_train)
dim(real_estate_test)

head(real_estate_train)


analysis_data$price_category <- as.factor(analysis_data$price_category)    

real_estate_train$price_category <- as.factor(real_estate_train$price_category)

#The tuneRF function in the randomForest package is used for tuning the parameter mtry, which represents 
#the number of features randomly chosen at each split in the tree-building process. It's an important parameter in Random Forest models, controlling the randomness and diversity of each tree in the forest.

bestmtry <- tuneRF(real_estate_train,real_estate_train$price_category,stepFactor = 1.2, improve = 0.01, trace=T, plot= T)


#CREATE RANDOM FOREST MODEL
model <- randomForest(price_category~.,data= real_estate_train)

model

#  # returns the importance of the variables : most siginificant - cp followed by thalach and so on......
importance(model) 

## visualizing the importance of variables of the model.
varImpPlot(model)


#MAKE PREDICTIONS ON THE TEST(VALIDATION) DATA 
pred_test <- predict(model, newdata = real_estate_test, type= "class")
pred_test

## The prediction to compute the confusion matrix and see the accuracy score 
confusionMatrix(table(pred_test,real_estate_test$price_category)) 


#NOW MAKE PREDICITON ON THE PREDICICTION DATA
test_predictions <- predict(model, newdata = prediction_data, type= "class")
test_predictions

#GETTING DESIRED OUTPUT FOR KAGGLE
result_df <- data.frame(id = prediction_data$id, price_category = test_predictions)

# Save the results to a CSV file
write.csv(result_df, file = "rf_model.csv", row.names = FALSE)



