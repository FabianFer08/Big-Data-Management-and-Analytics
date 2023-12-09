library(tidyverse)
library(tidymodels)

# Read the training dataset
analysis_data <- read.csv("data/dt_realestate_train_2022.csv", header = TRUE, sep= ',') 

prediction_data <- read.csv("data/dt_realestate_test_2022.csv", header = TRUE, sep= ',')


train_data$price_category  %>% table()

train_data$price_category  <- train_data$price_category  %>% factor()


real_estate_split <- initial_split(analysis_data, prop = 0.7)

real_estate_train <- training(real_estate_split)
real_estate_test <- testing(real_estate_split)

cv_folds <- real_estate_train %>% vfold_cv(v=10) #creating 10 folds from the training set

xgb_recipe <-
     recipe(price_category ~ . , data=real_estate_train)  %>% 
     #remove
     step_rm(id)  %>% 
     # convert to dummy variables 
     step_dummy(type_of_construction, energy_label)

xgb_model_tune <- 
   boost_tree(trees = tune(),
              tree_depth = tune(),
              learn_rate = tune(),
              stop_iter = 500)  %>%  #it stops after 500 iterations if model doesnt improve 
    set_mode("classification")  %>% 
    set_engine("xgboost")

xgb_tune_workflow <-
    workflow()  %>% 
    add_recipe(xgb_recipe)  %>% 
    add_model(xgb_model_tune)


class_metrics <- metric_set(accuracy, kap, sensitivity, specificity, roc_auc)

install.packages("doParallel")
library(doParallel)

num_cores <- parallel::detectCores()
doParallel::registerDoParallel(cores = num_cores - 1L) #So that the model runs faster

xgb_grid <- grid_max_entropy(trees(range = c(0,2000)),  #change after
                learn_rate(range =  c(-2, -1)),
                tree_depth(), size = 20)

install.packages('xgboost')
library(xgboost)


xgb_tune_result <- tune_grid(
    xgb_tune_workflow,
    resamples = cv_folds,
    grid = xgb_grid,
    metrics = class_metrics
)

xgb_tune_metrics <- 
  xgb_tune_result |>
  collect_metrics()

xgb_tune_metrics %>% subset(xgb_tune_metrics$.metric == 'accuracy') %>%  arrange(desc(mean))

xgb_best <- 
  xgb_tune_metrics |> 
  filter(tree_depth == 10, learn_rate == 0.0117, trees == 1428) |> 
  distinct(trees, tree_depth, learn_rate)

xgb_final_wf <- 
  finalize_workflow(xgb_tune_workflow, xgb_best)

xgb_final_fit <- 
  xgb_final_wf |>
  last_fit(real_estate_split, metrics = class_metrics)