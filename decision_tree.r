# Install and load necessary packages

library(rpart)
library(caret)

# Read the training dataset
train_data <- read.csv("data/dt_realestate_train_2022.csv", header = TRUE, sep= ',') 

test_data <- read.csv("data/dt_realestate_test_2022.csv", header = TRUE, sep= ',')



-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                                                         #Using CROSS-VALIDATION 
  
  
# how to set the hyper-parameters to increase the performance of the resulting tree as much as possible. To do that, we usually conduct cross-validation.  
  
# When dealing with classification problems, the best practice is to keep the ratio of different classes in each fold approximately the same as in the entire dataset.
  
  
  
    
# Replace spaces and other non-alphanumeric characters in factor levels with underscores
train_data$price_category <- make.names(train_data$price_category)                                                                                                        

# Set a seed for reproducibility
set.seed(123)  
# Define the levels for the target variable
price_levels <- c("< 100K", "100K-150K", "150K-200K", "200K-300K", "> 300K")

# Create a decision tree model using 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)
tree_model_cv <- train(price_category ~ ., data = train_data, method = "rpart", trControl = ctrl)  


# Access per-fold metrics
fold_metrics <- tree_model_cv$resample
print(fold_metrics)

# Make predictions on the test dataset using the trained model
test_predictions <- predict(tree_model_cv, test_data, type = "raw")
result_df <- data.frame(id = test_data$id, prediction = test_predictions)

# Export the result to a CSV file
write.csv(result_df, file = "results2.csv", row.names = FALSE)


#. The purpose of cross-validation is to assess how well the model generalizes to unseen data by evaluating it on
# multiple subsets of the training data.





----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                       #Saving a portion of the Train Data for testing.                                                                               
  
  
                                                                                                       
# Set a seed for reproducibility
set.seed(123)

# Split the training data into training and validation subsets
split_index <- createDataPartition(train_data$price_category, p = 0.7, list = FALSE)
train_subset <- train_data[split_index, ] #This is the subset of the training data used for training the model.
validation_subset <- train_data[-split_index, ]
# p = 0.8 means that 80% of the data will be used for training, and the remaining 20% will be used for validation.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Define the levels for the target variable
price_levels <- c("< 100K", "100K-150K", "150K-200K", "200K-300K", "> 300K")

# Create a decision tree model using the training subset
#tree_model <- rpart(price_category ~ ., data = train_subset, method = "class")

##SETTING SOME CONTROL PARAMETERS
tree_model <- rpart(price_category ~ ., data = train_subset, method = "class",
                    control = rpart.control(maxdepth = 11, cp = 0.01, minsplit = 10, minbucket = 5))



# (cp = 0.01, minsplit = 10, minbucket = 5)) Accuracy 0.5501706 
# 




#NOTES: 
#PRUNNING
# cp = 0.01: This means that the complexity parameter is set to 0.01. It's a relatively small value, indicating that the 
# tree will not be heavily pruned. A smaller cp allows the tree to capture more details and can lead to a more complex model.
# You can experiment with different values of cp to find a balance between model complexity and generalization. If you find that the model is too complex and 
# overfitting the training data, you might try increasing cp to encourage more pruning.

# SETTING MINIMUM NUMBER OF OBSERVATIONS REQUIRED TO ATTEMPT A SPLIT AT A NODE 
# minsplit = 10: This means that a node in the tree will only be considered for a split if it has 10 or more observations. If a node has fewer than 10 observations, 
# it will not be split further, and it becomes a terminal node. 
#  If minsplit is set too low, the tree may be prone to overfitting, capturing noise or outliers that are specific to the training data but do not generalize well to new data.
# On the other hand, setting minsplit too high may result in a too-coarse model, where the tree doesn't capture important patterns in the data.

#SETTING MINIMUM NUMBER OF OBSERVATIONS THAT MUST EXIST IN TERMINAL (LEAF) Node
#minbucket = 5: This means that a terminal node (leaf) in the tree must have at least 5 observations. If, after a split, a resulting node has fewer than 5 observations, the split is not performed,
#and the node becomes a terminal node.







# Make predictions on the validation subset
predictions <- predict(tree_model, validation_subset, type = "class")

# Convert actual values to factor with defined levels
validation_subset$price_category <- factor(validation_subset$price_category, levels = price_levels)

# Set the levels of the predicted variable to match the actual variable
predictions <- factor(predictions, levels = price_levels)

# Evaluate model performance on the validation subset
conf_matrix <- confusionMatrix(predictions, validation_subset$price_category)

# Print confusion matrix
print(conf_matrix)

# Print accuracy
accuracy <- conf_matrix$overall["Accuracy"]
cat("Accuracy:", accuracy, "\n")

---------------

test_predictions <- predict(tree_model, test_data, type = "class")
result_df <- data.frame(id = test_data$id, prediction = test_predictions)


write.csv(result_df, file = "first_model_dt.csv", row.names = FALSE)
