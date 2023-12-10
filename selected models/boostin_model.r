#Boosting is a sequential technique that works on the principle of ensemble. It combines a set of weak
# learners and delivers improved prediction accuracy. At any instant t, the model outcomes are weighed based 
# on the outcomes of the previous instant t-1. The outcomes predicted correctly are given a lower weight and 
#the ones miss-classified are weighted higher. This technique is followed for a classification problem while 
#a similar technique is used for regression.

library(tidyverse)
library(tidymodels)
library(gbm)

# Read the training dataset
analysis_data <- read.csv("data/dt_realestate_train_2022.csv", header = TRUE, sep= ',') 
prediction_data <- read.csv("data/dt_realestate_test_2022.csv", header = TRUE, sep= ',')

# Convert to factor variables
analysis_data$price_category <- as.factor(analysis_data$price_category)
analysis_data$energy_label <- factor(analysis_data$energy_label)
analysis_data$type_of_construction <- factor(analysis_data$type_of_construction)

prediction_data$price_category <- as.factor(prediction_data$price_category)
prediction_data$energy_label <- factor(prediction_data$energy_label)
prediction_data$type_of_construction <- factor(prediction_data$type_of_construction)

# Model assessment setup
real_estate_split <- initial_split(analysis_data, prop = 0.7)
real_estate_train <- training(real_estate_split)
real_estate_test <- testing(real_estate_split)

# Set up the boosting model for classification
best_tree_depth <- 10
best_learn_rate <- 0.0117
best_trees <- 1428

boost <- gbm(price_category ~ ., data = real_estate_train,
             n.trees = best_trees,
             shrinkage = best_learn_rate,
             distribution = "multinomial",  # Specify classification
             interaction.depth = best_tree_depth,
             bag.fraction = 0.7,
             n.minobsinnode = 5)

# Use the model to predict the price category of the test data
predictions <- predict(boost, newdata = prediction_data, type = "response", n.trees = best_trees)

# Convert numeric predictions to factor levels
predicted_categories <- colnames(predictions)[apply(predictions, 1, which.max)]
predicted_categories <- factor(predicted_categories, levels = levels(analysis_data$price_category))

# Assess accuracy on the test set
accuracy <- mean(predicted_categories == prediction_data$price_category)
cat("Accuracy on Test Set: ", accuracy, "\n")

# Create a data frame with id and predictions
result_df <- data.frame(id = prediction_data$id, predictions = predicted_categories)
write.csv(result_df, "boosting_model.csv", row.names = FALSE)
