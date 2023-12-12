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
best_tree_depth <- 15
best_learn_rate <- 0.0117
best_trees <- 1428

boost <- gbm(price_category ~ ., data = analysis_data,
             n.trees = best_trees,
             shrinkage = best_learn_rate,
             distribution = "multinomial",  # Specify classification
             interaction.depth = best_tree_depth,
             bag.fraction = 0.9, #0.6, 0.8, 0.9, -0.5-
             n.minobsinnode = 8) #5 #20 #10 #8 

#Display variable importance
boost_summary <- summary(boost)
print(boost_summary)

# Save the variable importance plot as a PNG file
png("variable_importance_plot.png", width = 800, height = 600, units = "px")
plot(boost_summary, main = "Variable Importance")
dev.off()


#Normalized to 1 
boost_summary <- summary(boost)
boost_summary$rel.inf <- boost_summary$rel.inf / sum(boost_summary$rel.inf)  # Normalize to 1
print(boost_summary)

# Save the normalized variable importance plot as a PNG file
png("normalized_variable_importance_plot.png", width = 800, height = 600, units = "px")
plot(boost_summary, main = "Normalized Variable Importance")
dev.off()


#See accuracy in training set to get confusion matrix, then try in the prediction data
predictions_test1 <- predict(boost, newdata = analysis_data, type = "response", n.trees = best_trees)

predicted_categories2 <- colnames(predictions_test1)[apply(predictions_test1, 1, which.max)]
predicted_categories2 <- factor(predicted_categories2, levels = levels(analysis_data$price_category))



# Use the model to predict the price category of the test data
predictions <- predict(boost, newdata = prediction_data, type = "response", n.trees = best_trees)

# Convert numeric predictions to factor levels
predicted_categories <- colnames(predictions)[apply(predictions, 1, which.max)]
predicted_categories <- factor(predicted_categories, levels = levels(analysis_data$price_category))


install.packages('caret')
library(caret)

confusionMatrix(table(predicted_categories2,analysis_data$price_category)) 
# Assess accuracy on the test set
accuracy <- mean(predicted_categories == prediction_data$price_category)
cat("Accuracy on Test Set: ", accuracy, "\n")

# Create a data frame with id and predictions
result_df <- data.frame(id = prediction_data$id, predictions = predicted_categories)
write.csv(result_df, "boosting_model12.csv", row.names = FALSE)
