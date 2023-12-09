---
title: "Random Forests on the Peer-to-Peer Lending Data"
date: "Compiled `r lubridate::now()`"
output:
  html_document:
    highlight: pygments
    theme: flatly
    df_print: paged
editor_options:
  chunk_output_type: console
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE,
											tidy = 'styler')
```


<style>
pre.sourceCode {
  background-color: #fcffff;
  color: #262c2d;
}
code span.at {
  color: #ca863c;
}
code span.dv {
  color: #047d39;
}
code span.fu {
  color: #7629d2;
}
code span.do {
  color: #999;
  font-style: normal;
}
code span.st {
  color: #cc4e11;
}
code span.sc {
  color: #ca7dac;
}
pre > code.sourceCode a {
  display: none;
}
blockquote > p {
  font-size: 13px !important;
  color: #647980;
  line-height: 1.5 !important;
  font-weight: 400 !important;
}
blockquote > code {
   rgba(0, 0, 0, 0.02);
}
details {
  border-left-style: outset;
  border-left-width: thick;
  padding-left: 10px;
  padding-bottom: 5px;
}
</style>



# Objectives

* Execute a basic exploratory data analysis in R  
* Apply random forests to data in R using *tidymodels*
* Understand how to compute and interpret variable importance measures
* Execute several visual methods for assessing the quality of class probability predictions

# Packages

Load packages:

```{r message = FALSE}
library(tidymodels)
library(tidyverse)
library(skimr)
```


# Data

The data for this tutorial are the same peer-to-peer lending data from the decision trees tutorial in Module 4.


```{r}
load("home_improv.RData")
```

The goal is to predict `loan_status`, and specifically, whether a loan will default (the positive case) or not. 


# Model assessment setup

We use the same model assessment setup as in the decision trees tutorial. Using the same random seed will allow us to compare our results to that of the other notebook.

We need to tune a random forest model, and will do this using K-fold cross-validation. To test the tuned model, we will retain some of the data. Hence, we start by making a stratified train-test split (recall the target variable is imbalanced), keeping 70% for training and 30% for testing. 

```{r}
set.seed(912340)
home_improv_split <- initial_split(data = home_improv, prop = 0.7, 
                          strata = loan_status)
```

The separate sets are then obtained:

```{r}
home_improv_train <- training(home_improv_split)
home_improv_test <- testing(home_improv_split)
```

Tuning will take place using 10-fold stratified CV:

```{r}
set.seed(82001)
cv_folds <- 
  home_improv_train |> 
  vfold_cv(v = 10, strata = loan_status)
```


# Exploratory data analysis

The main tools for exploratory data analysis---which is a crucially important step before modeling---are plotting and generating summary statistics. 

Start by printing summary statistics:

```{r}
skim(home_improv_train)
```

Notice the following:

* `emp_length` has a number of missing values (its `complete_rate` is `0.9`)
* `loan_amnt` and `funded_amnt` appear to have the same distributions
* `dti` may contain some outliers
* `fico_range_low` and `fico_range_high` have distributions that differ by four points at all percentiles shown except for the maximum.

Because `emp_length` has missing values, we need to decide what to do with this variable before training a random forest. Possibilities include:

* Omit the variable,
* Omit the rows containing the missing values, and
* Impute the missing values. 

Imputation can be done in a variety of ways; various possibilities are available as functions in the **recipes** package, such as `step_kknimpute()` and `step_modeimpute()`.  For this tutorial, we will omit the variable, as it is the simplest solution. 

Due to nearly perfect collinearity, only one of `loan_amnt` and `funded_amnt`, and only one of `fico_range_low` and `fico_range_high` should be included. 

Plot the correlation matrix to confirm, and potentially identify other problems: 

```{r }
home_improv_train |> 
  select_if(is.numeric) |> 
  cor() |> 
  corrplot::corrplot()
```

The previous two steps generated univariate (via `skim()`) summaries and pairwise summarise (via `corrplot()`). For supervised learning, another important exploratory step (using the training data only!) is understanding the relationship between features and the target variable. Following are a few examples of this.

To explore relationships between numeric features and the target variable, one can group the data and *then* apply `skim()`. This produces summaries of *conditional* distributions of feature variables given each value of the target variable. 

Here are the results for the numeric variables (after omitting variables that will not be used for training), extracted using `yank()`:

```{r}
home_improv_train |> 
  group_by(loan_status) |> 
  skim(-id, -funded_amnt, -fico_range_low) |> 
  yank("numeric")
```

The most promising features for performing classification are those for which the conditional distributions (with respect to `loan_status`) differ most. The coarse histograms in the final column clearly identify `int_rate`, `dti`, `annual_inc`, and `recoveries` (among others) as promising features. However, we cannot use `recoveries` because it will introduce data leakage (see the decision tree tutorial for more detail). Also note that `total_paymnt`, which is also a leaky variable, does not really stand out as problematic the way `recoveries` does. Domain knowledge is needed to decide whether variables should or should not be included as predictors. 

Exploratory data analysis can help to identify leaky variables, but is not sufficient to discover all leaky variables. There is no substitute for situational fluency---that is, having a deep understanding the data, use case, and broader business context.

Below are a few more plots showing some relations. The outliers of `dti` occur for fully paid loans:

```{r echo = FALSE}
home_improv_train |> 
  ggplot(aes(x = loan_status, y = dti)) +
  geom_boxplot() + 
  scale_y_sqrt() +
  theme_bw()
```

The conditional distribution of `loan_status` covaries with `grade`, such that defaults are more likely for lower grade loans (A = best grade, G = worst):

```{r echo = FALSE}
home_improv_train |> 
  ggplot(aes(x = grade, fill = loan_status)) +
  geom_bar(position = "fill") +
  scale_fill_viridis_d(option = 'E', direction = -1, end = .8) +
  theme_bw() 
```

Interest rates are discrete and appear to be set based on values of `grade`. The function `geom_density_ridges()` from **ggridges** reveals the following:

```{r}
home_improv_train |> 
  ggplot(aes(x = int_rate, y = grade)) + 
  ggridges::geom_density_ridges(bandwidth = 0.1) +
  theme_bw()
```

Loans to lenders with lower FICO credit ratings are more likely to default:

```{r echo = FALSE}
home_improv_train |> 
  ggplot(aes(x = fico_range_high, fill = loan_status)) + 
  geom_density(alpha = 0.5) +
  scale_fill_viridis_d(option = 'D') +
  theme_bw()
```

Based on the exploratory analysis:

* `emp_length` will be excluded due to missing values
* `funded_amnt` and `fico_range_low` will be excluded due to collinearity
* `recoveries` and `total_paymnt` will be excluded to avoid data leakage


# Defining the random forest model

Next, build a random forest classifier using **tidymodels**. The **ranger** package will be the model engine.

Define a pre-processing recipe:

* Excludes the variables mentioned above
* Removes `id`, as it is meta-data
* Performs down-sampling for class imbalance, as in the previous notebook, using `themis::step_downsample()`.

The "excluded" variables will be retained, but not used for modelling, by changing their roles to "metadata":
  
```{r}
rf_recipe_downsample <- 
  recipe(loan_status ~ ., data = home_improv_train) |> 
  update_role(emp_length, id, total_pymnt, recoveries, 
              funded_amnt, fico_range_low, 
              new_role = "metadata") |> 
  themis::step_downsample(loan_status)
```

Print the recipe:

```{r eval = FALSE}
rf_recipe_downsample
```

```{r echo = FALSE}
print_recipe_for_rmarkdown <- function(recipe) {
  cli::cli_fmt(recipe |> print()) |> 
  stringr::str_replace("^$", " ") |> 
  cat(sep = "\n")
}
print_recipe_for_rmarkdown(rf_recipe_downsample)
```
 
Define the random forest model. The number of trees is set to 1000. The value of `mtry`, which determines the number of predictors that are randomly sampled at each split, will be tuned. Adding `importance = "permutation"` to the engine definition allows us to evalulate variable importances, which will happen later in the tutorial.

```{r}
rf_model_tune <- 
  rand_forest(mtry = tune(), trees = 1000) |>
  set_mode("classification") |>
  set_engine("ranger", importance = "permutation")
```

Create the workflow object:

```{r}
rf_tune_wf <-
  workflow() |>
  add_recipe(rf_recipe_downsample) |>
  add_model(rf_model_tune)
rf_tune_wf
```

The metric set is the same as in the decision tree tutorial:

```{r}
class_metrics <- metric_set(accuracy, kap, sensitivity, 
                            specificity, roc_auc)
```

# Tuning the model 

There are 12 features in the recipe, so the value of `mtry` will be tuned on values from 1 to 12. 

```{r}
rf_tune_grid <- grid_regular(mtry(range = c(1, 12)), levels = 12)
```

Use parallel processing to speed up computation:

```{r}
num_cores <- parallel::detectCores()
num_cores
doParallel::registerDoParallel(cores = num_cores - 1L)
```

Tune the model. This may take a few minutes to complete. You can save time by using fewer trees (e.g., 500).

```{r}
set.seed(99154345)
rf_tune_res <- tune_grid(
  rf_tune_wf,
  resamples = cv_folds,
  grid = rf_tune_grid,
  metrics = class_metrics
)
```

Recall from the decision tree tutorial that sensitivity is most important, and specificity is also important.

Plot results for these metrics:

```{r}
rf_tune_res |>
  collect_metrics() |>
  filter(.metric %in% c("sensitivity", "specificity")) |>
  ggplot(aes(x = mtry, y = mean, ymin = mean - std_err,
             ymax = mean + std_err, 
             colour = .metric)) +
  geom_errorbar() + 
  geom_line() +
  geom_point() +
  scale_colour_manual(values = c("#D55E00", "#0072B2")) +
  facet_wrap(~.metric, ncol = 1, scales = "free_y") +
  guides(colour = 'none') +
  theme_bw()
```

There is not much difference in performance for the values of `mtry` that we investigated, especially not when taking into accound the standard errors.

The results for AUC, accuracy and Cohen's Kappa tell a similar story:

```{r}
rf_tune_res |>
  collect_metrics() |>
  filter(.metric %in% c("roc_auc", "accuracy", "kap")) |>
  ggplot(aes(x = mtry, y = mean, ymin = mean - std_err, 
             ymax = mean + std_err, colour = .metric)) +
  geom_errorbar() + 
  geom_line() +
  geom_point() +
  scale_colour_manual(values = c("#D55E00", "#0072B2", "#009E73")) +
  facet_wrap(~.metric, ncol = 1, scales = "free_y") +
  guides(colour = 'none') +
  theme_bw()
```

Select the best model using sensitivity:

```{r}
best_rf <- select_best(rf_tune_res, "sensitivity")
rf_final_wf <- finalize_workflow(rf_tune_wf, best_rf)
rf_final_wf
```

# Test set performance

Now the model has been tuned, train the finalized workflow on the entire training set and predict the test set:

```{r}
set.seed(9923)
rf_final_fit <- 
  rf_final_wf |>
  last_fit(home_improv_split, metrics = class_metrics)
```

The results on the test set for class predictions are:

```{r}
rf_final_fit |>
  collect_metrics()
```

Is this much better than our pruned classification tree from Module 4? The classification tree achieved a sensitivity and specificity of 0.629 and 0.648 on the test set, respectively. The random forest improves on sensitivity but is worse on specificity. Hence the random forest model does a better job on the goal of maximizing sensitivity. 

# Confusion matrix and visual assessments of performance

A confusion matrix for the test set predictions are as follows:

```{r}
rf_final_fit |> 
  collect_predictions() |> 
  conf_mat(truth = loan_status, estimate = .pred_class) 
```

As one would expect, there are many more false positives (top right) than false negatives (bottom left).

Here are some visualizations for the test set predictions. The ROC-curve can be constructed as follows:

```{r echo = FALSE, eval = FALSE, include = FALSE}
# make your own ROC curve...
rf_final_fit |> 
  collect_predictions() |> 
  # the next line is key: it arranges predictions in descending order based on
  # the probability of being assigned to the positive class
  arrange(desc(.pred_Default)) |> 
  mutate(sens = cumsum(loan_status == 'Default')/sum(loan_status == 'Default'),
         fpr = cumsum(loan_status != 'Default')/sum(loan_status != 'Default')) |> 
  ggplot() +
  aes(x = fpr, y = sens) + 
  geom_step() +
  geom_abline(linetype = 3) +
  labs(x = '1 - specificity', y = 'sensitivity') +
  theme_bw() +
  coord_equal()
```

```{r}
rf_final_fit |> 
  collect_predictions() |> 
  roc_curve(loan_status, .pred_Default) |> 
  autoplot()
```

There is room for improvement in terms of AUC. 

Plot the lift curve:

```{r}
rf_final_fit |> 
  collect_predictions() |> 
  lift_curve(loan_status, .pred_Default) |> 
  autoplot()
```

When targeting the 25% of loans with the highest predicted probability of default, a bit less than twice as many loans that defaulted are identified, compared to when targeting is done randomly.

Plot the gain curve:

```{r}
rf_final_fit |> 
  collect_predictions() |> 
  gain_curve(loan_status, .pred_Default) |> 
  autoplot()
```

When targeting the top 25% using this model, a little less than 50% of loans that will default would be uncovered. Note that the shaded region indicates the performance of an ideal (or "oracle") model, where all the loans that defaulted are ranked before the ones that ended up not defaulting. The ideal gain chart follows the left and tope edges of the shaded region.

# Variable importance scores

<!-- If we want variable importance measures, we need to do some extra work. We need to refit the model after specifying that we want variable importance to be computed as well. -->

<!-- The most effort goes into setting up a workflow with a new model specification which explicitly requests permutation variable importance. Here we hard code the selected value of `mtry`: -->

<!-- ```{r} -->
<!-- rf_model_vi <- rand_forest(mtry = best_acc$mtry, trees = 1000) |> -->
<!--   set_mode("classification") |> -->
<!--   set_engine("ranger", importance = "permutation") -->
<!-- rf_vi_wf <- workflow() |>  -->
<!--   add_model(rf_model_vi) |>  -->
<!--   add_recipe(rf_recipe_downsample) -->
<!-- ``` -->

<!-- (Note that the value mtry) -->

<!-- Now we can fit the model again, using the same random seed as above (when we used `last_fit()`): -->

<!-- ```{r} -->
<!-- set.seed(9923) -->
<!-- rf_vi_fit <- rf_vi_wf |> fit(data = home_improv_train) -->
<!-- ``` -->

The **vip** package provides functions `vi()` for extracting the variable importances and `vip()` for plotting them:

```{r}
rf_final_fit |> 
  extract_fit_parsnip() |> 
  vip::vi()
rf_final_fit |> 
  extract_fit_parsnip() |> 
  vip::vip(geom = "point", num_features = 12) +
  theme_bw()
```

`int_rate` is the most important feature, followed by `grade`, `fico_range_high`, and `loan_amnt`.

# Making further improvements

The estimated random forest performed better than the single classification tree (from the tutorial in Module 4) on the test data. The random forest model could be further improved:

* Adding more observations. This should be the first priority, as it is likely to bring the most reward for the effort needed.
* Adding more features. Rather than dropping variables with missing data, these might instead be imputed.
* Engineering new features from existing, measured variables. What could cause lenders to default on their loans? If our intuition suggests a factor that isn't already expressed in the data as a single variable, then we can create a new feature that better reflects that factor. This will allow the model to "see" this factor more clearly. This approach can be very rewarding, but it requires a deep consideration of the mechanisms behind loan default and how they could be quantified using the data at hand.
* Performing a careful error analysis, which looks at the errors that the model are making and how they relate to the features. This can provide insight that can be used to improve the mode. An error analysis is also relevant for ethical and fairness purposes, when it is important that the model should perform similarly for different sub-populations (of lenders, in this case). Remember that the metrics we used here and elsewhere are estimates of the average generalization error. There may very well be subgroups for which the model performs poorly. In practice, this knowledge can be extremely important.


# R session information

```{r}
sessionInfo()
```

