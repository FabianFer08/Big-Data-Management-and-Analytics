library(dplyr)       #to perform some data wrangling tasks.
library(rpart)       #to fit decision trees without tuning.
library(rpart.plot)  #to plot our decision trees.
library(Metrics)     #to assess the performance of our models;
library(mlr)         #to train our model’s hyperparameters.
library(ggplot2)     #for general plots we will do.
library(plotly)      #for 3-D plots.

train_data <- read.csv("dt_realestate_train_2022.csv", header = TRUE, sep= ';') 

test_data <- read.csv("dt_realestate_test_2022.csv", header = TRUE, sep= ';')


#Let’s also split our data into train an test — leaving 20% of the data as an holdout group:

# Set a seed for reproducibility
set.seed(123)

# Split the training data into training and validation subsets
split_index <- createDataPartition(train_data$price_category, p = 0.8, list = FALSE) #leaving 20% of the data as a holdout group. 
train_subset <- train_data[split_index, ] #This is the subset of the training data used for training the model.
validation_subset <- train_data[-split_index, ]

# Define the levels for the target variable
price_levels <- c("< 100K", "100K-150K", "150K-200K", "200K-300K", "> 300K")

# Create a decision tree model using the training subset
#tree_model <- rpart(price_category ~ ., data = train_subset, method = "class")


tree_model <- rpart(price_category ~ ., data = train_subset, method = "class") #Using default hyperparameters

#maxdepth = 30 minsplit = 20 ; 7 — minbucket = 7; The split must increase the “performance” (although it’s not that direct, we can consider “performance” a proxy for cp) of the tree by at least 0.01 — cp = 0.01 ;


predictions <- predict(tree_model, validation_subset, type = "class")

accuracy(validation_subset$price_category, predictions)  #To get accuracy could have skipped some steps in first notebook


#Now setting hyperparameters manually 
tree_model_custom <- rpart(price_category ~ ., data = train_subset, method = "class",
                    control = rpart.control(maxdepht = 5, cp = 0.001))


predictions2 <- predict(tree_model_custom, validation_subset, type = "class")
accuracy(validation_subset$price_category, predictions2)



# There is a combination of hyperparameter that lead to optimum performance on the test set, the library mlr allows us to find those. 

# Hyperparameter Tuning using MLR — Tweaking one Parameter

#The mlr library uses exactly the same method we will learn to tweak parameters for random forests, xgboosts, SVM’s, etc.


# Although caret (Used before) also has some built-in hyperpameter search, mlr enable us to view the impact of those hyperparameters much better, being less “black-boxy”.

# As we’ve discussed, one of the advantages is that it let us view each hyperparameter impact on the performance of the model.

getParamSet("classif.rpart") # returns all the tweakable parameters available for a specific model


#ie: Based on Max depht, mlr fit these 30 different versions of the decision tree and evaluate the accuracy of those models.

train_subset$type_of_construction <- as.factor(train_subset$type_of_construction)
train_subset$energy_label <- as.factor(train_subset$energy_label)

d.tree.params <- makeClassifTask(
  data=train_subset, 
  target="price_category")



param_grid <- makeParamSet( 
 makeDiscreteParam("maxdepth", values=1:30))
 
#What I’m stating in the code above is that my tree will iterate on 30 different values of maxdepth, a vector (1:30) that contains 1, 2, 3 … , 30 as the values to feed into the hyperparamter.


# Define Grid
control_grid = makeTuneControlGrid()
# Define Cross Validation
resample = makeResampleDesc('CV', iters = 3L) #THREE FOLD CROSS VALIDATION 
# Define Measure
measure = acc


#All set ! Time to feed everything into the magicaltuneParams function that will kickstart our hyperparameter tuning!


set.seed(123)
dt_tuneparam <- tuneParams(learner='classif.rpart', 
                           task=d.tree.params, 
                           resampling = resample,
                           measures = measure,
                           par.set=param_grid, 
                           control=control_grid, 
                           show.info = TRUE)

#As you run the code above, our hyperparameter search will start to execute! show.info = TRUE will output the execution’s feedback:

#Each maxdepthis generating an acc.test.mean, a mean of the accacross the several datasets used in the Cross Validation. mlralso let us evaluate the results using generateHyperParsEffectData :


result_hyperparam <- generateHyperParsEffectData(dt_tuneparam, partial.dep = TRUE)

#And we can plot the evolution of our accuracy using

ggplot(
  data = result_hyperparam$data,
  aes(x = maxdepth, y=acc.test.mean)
) + geom_line(color = 'darkblue')

#Looking at our plot, we understand that after a deepness of 5, the effects on the accuracy are marginal with infinitesimal differences.

#WE CAN CONFRIM THE BEST MODEL CHOSEN BY THE TuneParams function by calling: 

dt_tuneparam

#nevertheless, let’s fit our best parameters using the object dt_tuneparam$x to pick up the saved hyperparameters and store them usingsetHyperPars :


best_parameters = setHyperPars(
  makeLearner("classif.rpart"), 
  par.vals = dt_tuneparam$x
)
best_model = train(best_parameters, d.tree.params)

#train will fit a decision tree with the saved hyperparameters in the best_parameters object,
# After running the code above, we have a fitted tree with the best hyperparameters returned from the grid search on best_model.




#To evaluate this model on our test set, we need to make a new makeClassifTask pointing to the test data:
validation_subset$type_of_construction <- as.factor(validation_subset$type_of_construction)
validation_subset$energy_label <- as.factor(validation_subset$energy_label)

d.tree.mlr.test <-makeClassifTask(
  data=validation_subset, 
  target="price_category")

results <- predict(best_model, task = d.tree.mlr.test)$data
accuracy(results$truth, results$response)










#----------------------------------------------------------------------------------------------------------------------------------------------#



#With mlr , we can tweak the entire landscape of parameters at the same time and with just a small tweak in our code!
getParamSet("classif.rpart")

param_grid_multi <- makeParamSet( 
  makeDiscreteParam("maxdepth", values=1:20),
 # makeNumericParam("cp", lower = 0.001, upper = 0.01),
  makeDiscreteParam("minsplit", values=1:30)
)


set.seed(123)
dt_tuneparam_multi <- tuneParams(learner='classif.rpart', 
                           task=d.tree.params, 
                           resampling = resample,
                           measures = measure,
                           par.set=param_grid_multi, 
                           control=control_grid, 
                           show.info = TRUE)

# You’ll notice that the dt_tuneparam_multi will take more time than the dt_tuneparam search because we will be fitting near 3000(!) trees to our data.
# In the [Tune] output, we have the best parameters for our search:

#setting task
d.tree.mlr.train <- makeClassifTask(
  data=train_subset, 
  target="price_category")

# Extracting best Parameters from Multi Search
best_parameters_multi = setHyperPars(
  makeLearner(“classif.rpart”, predict.type = “prob”), 
  par.vals = dt_tuneparam_multi$x
  
best_model_multi = train(best_parameters_multi, d.tree.mlr.train)

d.tree.mlr.test <-makeClassifTask(
  data=validation_subset, 
  target="price_category")

results <- predict(best_model_multi, task = d.tree.mlr.test)$data
accuracy(results$truth, results$response)