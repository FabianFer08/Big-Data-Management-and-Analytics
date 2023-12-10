library(dplyr)
library(rpart)
library(rpart.plot)
library(Metrics)
library(mlr)
library(ggplot2)
library(plotly)
library(caret)

# Read the training dataset
analysis_data <- read.csv("data/dt_realestate_train_2022.csv", header = TRUE, sep= ',') 

prediction_data <- read.csv("data/dt_realestate_test_2022.csv", header = TRUE, sep= ',')



real_estate_split <- initial_split(analysis_data, prop = 0.7)

real_estate_train <- training(real_estate_split)
real_estate_test <- testing(real_estate_split)



#checking that levels are correctly ordered in the data 
levels(analysis_data$price_category)


#if not, use the factor fucntion
real_estate_test$price_category <- factor(real_estate_test$price_category, levels = c("< 100K", "100K-150K", "150K-200K", "200K-300K", "> 300K"))

d.tree <- rpart(price_category ~ ., data = real_estate_train, method = "class") 

#lets check the accuracy on the test set
predicted_values <- predict(d.tree, real_estate_test, type = "class")
predicted_values <- factor(predicted_values, levels = c("< 100K", "100K-150K", "150K-200K", "200K-300K", "> 300K"))

accuracy(real_estate_test$price_category, predicted_values)


#This should provide you with the accuracy value. The confusionMatrix function is more flexible and can handle different types of inputs.
# Create a confusion matrix
conf_matrix <- confusionMatrix(data = predicted_values, reference = real_estate_test$price_category)
accuracy_value <- conf_matrix$overall["Accuracy"]
print(accuracy_value)

#Now setting hyperparameters manually 
tree_model_custom <- rpart(price_category ~ ., data = real_estate_train, method = "class",
                    control = rpart.control(maxdepht = 5, cp = 0.001))

# mlr,the Machine Learning in R Library is a cool artificial intelligence package in R that gives us the tools to train several different 
# models and perform tuning. As we’ve discussed, one of the advantages is that it let us view each hyperparameter impact on the performance of the model.

install.packages("XML") #And select No when installing
getParamSet("classif.rpart")

real_estate_train$type_of_construction <- as.factor(real_estate_train$type_of_construction)
real_estate_train$energy_label <- as.factor(real_estate_train$energy_label)

#Defining classification Task
d.tree.params <- makeClassifTask(
  data=real_estate_train, 
  target="price_category")

#we need to create a grid of parameters to iterate on 
param_grid <- makeParamSet( 
 makeDiscreteParam("maxdepth", values=1:30))

 # Define Grid
control_grid = makeTuneControlGrid()
# Define Cross Validation
resample = makeResampleDesc('CV', iters = 3L) #THREE FOLD CROSS VALIDATION 
# Define Measure
measure = acc


# All set ! Time to feed everything into the magicaltuneParams function that will kickstart our hyperparameter tuning!

install.packages("XML") #And select No when installing
set.seed(123)
dt_tuneparam <- tuneParams(learner='classif.rpart', 
                           task=d.tree.params, 
                           resampling = resample,
                           measures = measure,
                           par.set=param_grid, 
                           control=control_grid, 
                           show.info = TRUE)

# Result: maxdepth=7 : acc.test.mean=0.5708843, could also be 5-6 as no much difference in accuracy 

result_hyperparam <- generateHyperParsEffectData(dt_tuneparam, partial.dep = TRUE)

#And we can plot the evolution of our accuracy using


plot <- ggplot(
  data = result_hyperparam$data,
  aes(x = maxdepth, y = acc.test.mean)
) + geom_line(color = 'darkblue')

# Save the plot as a PNG file

ggsave("accuracy_plot.png", plot, width = 8, height = 6, units = "in")

#WE CAN CONFRIM THE BEST MODEL CHOSEN BY THE TuneParams function by calling: 

dt_tuneparam

#let’s fit our best parameters using the object dt_tuneparam$x to pick up the saved hyperparameters and store them usingsetHyperPars :

best_parameters = setHyperPars(
  makeLearner("classif.rpart"), 
  par.vals = dt_tuneparam$x
)

#created the classification task using d.tree.params. Fit tree with the best hyperparameters returned from the grid search
best_model = train(best_parameters, d.tree.params)



#EVALUATING IN TEST SET

real_estate_test$type_of_construction <- as.factor(real_estate_test$type_of_construction)
real_estate_test$energy_label <- as.factor(real_estate_test$energy_label)

d.tree.task.test <- makeClassifTask(
 data=real_estate_test, 
 target="price_category"
)

results <- predict(best_model, task = d.tree.task.test)$data
#accuracy(results$truth, results$response)

#You can use the confusionMatrix function from the caret package to calculate accuracy. Here's how you can do it:

# Convert the factor levels to match between truth and response
results$truth <- factor(results$truth, levels = levels(results$response))

# Create a confusion matrix
conf_matrix <- confusionMatrix(data = results$response, reference = results$truth)

# Extract accuracy from the confusion matrix
accuracy_value <- conf_matrix$overall["Accuracy"]
print(accuracy_value)


#GRAPH WITH ACTUAL AND PREDICTED NUMBER PER CATEGORY. 

library(ggplot2)

# Combine the actual and predicted data for plotting
plot_data <- data.frame(
  Category = rep(levels(results$truth), each = 2),
  Value = c(table(results$truth), table(results$response)),
  Type = rep(c("Actual", "Predicted"), times = length(levels(results$truth)))
)

# Create a bar chart
my_plot <- ggplot(plot_data, aes(x = Category, y = Value, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7, color = "white") +
  labs(title = "Actual vs. Predicted Counts by Category",
       x = "Category",
       y = "Count") +
  scale_fill_manual(values = c("Actual" = "blue", "Predicted" = "orange")) +
  theme_minimal()

  ggsave("actual_vs_predicted_counts.png", my_plot, width = 8, height = 6, units = "in")



# Tweaking Multiple Parameters ###############################################################################

param_grid_multi <- makeParamSet( 
  makeDiscreteParam("maxdepth", values=1:20),
  makeNumericParam("cp", lower = 0.001, upper = 0.01),
  makeDiscreteParam("minsplit", values=1:30)
)

#And how can we train this multi-parameter search? By feeding our param_grid_multi to the tuneParams function!

set.seed(123)
dt_tuneparam_multi <- tuneParams(learner='classif.rpart', 
                           task=d.tree.params, 
                           resampling = resample,
                           measures = measure,
                           par.set=param_grid_multi, 
                           control=control_grid, 
                           show.info = TRUE)

#This combination of hyperparameters yielded an accuracy of around XX% on the cross-validation
#Let’s extract the best parameters, train a new tree with them and see the result on our test set:
# Result: maxdepth=10; cp=0.001; minsplit=3 : acc.test.mean=0.6258373

result_hyperparam <- generateHyperParsEffectData(dt_tuneparam_multi, partial.dep = TRUE)

#And we can plot the evolution of our accuracy using


plot <- ggplot(
  data = result_hyperparam$data,
  aes(x = maxdepth, y = acc.test.mean)
) + geom_line(color = 'darkblue')

# Save the plot as a PNG file

ggsave("accuracy_plot_multi.png", plot, width = 8, height = 6, units = "in")







# Extracting best Parameters from Multi Search
best_parameters_multi = setHyperPars(
 makeLearner("classif.rpart" , predict.type = 'prob'), 
 par.vals = dt_tuneparam_multi$x
)


best_model_multi = train(best_parameters_multi, d.tree.params)

# Predicting the best Model
results2 <- predict(best_model_multi, task = d.tree.task.test)$data

# Convert the factor levels to match between truth and response
results2$truth <- factor(results2$truth, levels = levels(results2$response))

# Create a confusion matrix
conf_matrix <- confusionMatrix(data = results2$response, reference = results2$truth)

# Extract accuracy from the confusion matrix
accuracy_value <- conf_matrix$overall["Accuracy"]
print(accuracy_value)

# -------------- Getting Output ------------------ #



# Predicting using the best_model obtained from the tunning CV = 3 FOLDS
test_predictions <- predict(best_model_multi, newdata = prediction_data, type = "class")

# Assuming 'id' is a column in the 'prediction_data' data frame
result_df <- data.frame(id = prediction_data$id, price_category = test_predictions)

# Save the results to a CSV file
write.csv(result_df, file = "v2_d_tree_tunned_3FOLDS.csv", row.names = FALSE)






### -------------------------- ### -------------------------- ### -------------------------- ### -------------------------- ###

# Get variable importance measures
variable_importance <- getFeatureImportance(best_model_multi)

# Print the variable importance
print(variable_importance)

str(variable_importance)




# Extracting importance values and variable names
importance_values <- variable_importance$res$importance
variable_names <- variable_importance$res$variable

# Combine into a data frame
importance_df <- data.frame(Variable = variable_names, Importance = importance_values)

# Print the variable importance
print(importance_df)

# Plot variable importance using dotplot
library(lattice)

# Create a dotplot
plot2 <- dotplot(Importance ~ Variable, data = importance_df,
                 main = "Var Importance Plot",
                 xlab = "Importance", ylab = "Variable",
                 col = "skyblue", cex = 0.7,
                 scales = list(x = list(rot = 45, cex = 0.8)))

png("variable_importance_plot.png", width = 800, height = 600)



dev.off()
print(plot2)

