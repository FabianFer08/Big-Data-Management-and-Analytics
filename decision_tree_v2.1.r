
library(tidyverse)
library(tidymodels)

library(rpart)
library(caret)

# Read the training dataset

analysis_data <- read.csv("data/dt_realestate_train_2022.csv", header = TRUE, sep= ',') 

prediction_data <- read.csv("data/dt_realestate_test_2022.csv", header = TRUE, sep= ',')


# train_data$price_category  %>% table()


# train_data$price_category  <- train_data$price_category  %>% factor()


# Model assessment setup

real_estate_split <- initial_split(analysis_data, prop = 0.7)

real_estate_train <- training(real_estate_split)
real_estate_test <- testing(real_estate_split)


# Define the levels for the target variable
price_levels <- c("< 100K", "100K-150K", "150K-200K", "200K-300K", "> 300K")

##SETTING SOME CONTROL PARAMETERS
tree_model <- rpart(price_category ~ ., data = real_estate_train, method = "class",
                    control = rpart.control(maxdepth = 11, cp = 0.01, minsplit = 10, minbucket = 5))


# Make predictions on the validation subset
predictions <- predict(tree_model, real_estate_test, type = "class")

# Convert actual values to factor with defined levels
real_estate_test$price_category <- factor(real_estate_test$price_category, levels = price_levels)

# Set the levels of the predicted variable to match the actual variable
predictions <- factor(predictions, levels = price_levels)

# Evaluate model performance on the validation subset
conf_matrix <- confusionMatrix(predictions, real_estate_test$price_category)

# Print confusion matrix
print(conf_matrix)

# Print accuracy
accuracy <- conf_matrix$overall["Accuracy"]
cat("Accuracy:", accuracy, "\n")

---------------

test_predictions <- predict(tree_model, prediction_data, type = "class")
result_df <- data.frame(id = prediction_data$id, prediction = test_predictions)


write.csv(result_df, file = "v2.1_model_dt.csv", row.names = FALSE)
                   