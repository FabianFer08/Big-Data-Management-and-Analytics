---
title: "Gradient Boosting Machines on the Peer-to-Peer Lending Data"
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

 * Apply gradient boosted decision trees to data in R

# Packages

Load 'em:

```{r message = FALSE}
library(tidymodels)
library(tidyverse)
```


# Data

We use again here the same peer-to-peer lending used in the previous tutorial, as well as the same model assessment setup.

Load the prepared data:

```{r}
load("home_improv.RData")
```

# Model assessment setup

Same as the previous tutorial:

```{r}
set.seed(912340)
home_improv_split <- initial_split(data = home_improv, prop = 0.7, 
                          strata = loan_status)

home_improv_train <- training(home_improv_split)
home_improv_test <- testing(home_improv_split)

set.seed(82001)
cv_folds <- home_improv_train |> vfold_cv(v = 10, strata = loan_status)
```

As we will see below, tuning will take some time. You could choose 5-fold CV instead to save time.

# Gradient boosting using **tidymodels**

We will use the **xgboost** package to fit the gradient boosted decision tree. It is worth taking a look at `?xgboost::xgb.train` for more information. As you will see, there are many parameters apart from the ones considered below that could be tuned.

The pre-processing recipe is similar to the one used for the random forest model (in the previous tutorial), with the difference that we must convert all nominal variables to dummies. This is because **xgboost** requires all variables to be numeric:
  
```{r}
xgb_recipe <- 
  recipe(loan_status ~ ., data = home_improv_train) |> 
  # remove these variables:
  step_rm(id, total_pymnt, recoveries, emp_length, funded_amnt, fico_range_low) |> 
  # convert to dummy variables:
  step_dummy(grade, home_ownership, verification_status, one_hot = TRUE) |> 
  themis::step_downsample(loan_status) 
```

Note the argument `one_hot = TRUE` passed to `step_dummy()`. This creates one dummy variable for each of the $K$ categories for a categorical variable, instead of having $K - 1$ dummy variables. This is a more useful representation for decision trees, as it allows all categories to be split off from the others in one split.

There are many hyper-parameters that can be tuned for a boosted tree model. Here we will tune the main three: the size of the ensemble (i.e., the number of trees), the depth of each tree, and the learning rate.

```{r}
xgb_model_tune <- 
  boost_tree(trees = tune(), 
             tree_depth = tune(), 
             learn_rate = tune(), 
             stop_iter = 500) |>
  set_mode("classification") |>
  set_engine("xgboost")
```

We have set `stop_iter = 500` to apply early stopping. If the model does not improve after an additional 500 iterations, training will stop and the code will move on to the next combination of tuning parameters. This can save a lot of time, because for larger learning rates many fewer iterations are typically required. Do not set this value too low. If computation time is not an issue, it is better not to set this value at all.

Define the workflow:

```{r}
xgb_tune_wf <- 
  workflow() |>
  add_recipe(xgb_recipe) |>
  add_model(xgb_model_tune)
xgb_tune_wf
```

The metric set is the same as in the previous tutorial:

```{r}
class_metrics <- metric_set(accuracy, kap, sensitivity, 
                            specificity, roc_auc)
```

Use parallel processing to speed up computation:

```{r}
num_cores <- parallel::detectCores()
num_cores
doParallel::registerDoParallel(cores = num_cores - 1L)
```

We can specify the tuning grid directly, or we can have `tune_grid()` guess on our behalf. We can also save time by performing a random search instead of a grid search. In a random search, not all combinations of tuning parameters are considered, which can save a lot of time. The `grid_random()`, `grid_latin_hypercube()` and `grid_max_entropy()` functions can help set up such a tuning grid. Here is an example of how to use these:

```{r}
set.seed(8504)
grid_max_entropy(trees(range = c(0, 10000)), 
                 learn_rate(range = c(-2, -1)), 
                 tree_depth(), size = 20)
```

It might be better to add the number of trees after determining values for `learn_rate` and `tree_depth`. Not all the auto-generated values might be appropriate (specifically, `tree_depth` has a large range).

For this tutorial, we will use the following grid:

```{r}
xgb_grid <- crossing(trees = 500 * 1:20, 
                     learn_rate = c(0.1, 0.01, 0.001), 
                     tree_depth = c(1, 2, 3))
xgb_grid
```

You may want to change the number of trees to `500 * 1:10` or less to save computing time at the first run. You can also save time by using fewer CV folds (perhaps 5).

We have used only three learning rates to save computation time. Note too that for trees, the maximum ensemble size is most important. Intermediate values will not result in retraining the model specifically for that value. For example, if the largest ensemble we consider contains 5000 trees, the result for 1000 trees will be derived using the first 1000 trees out of this same ensemble of 5000. A separate ensemble will not be created specifically for the case with 1000 trees. Here we have asked for the evaluation metrics to be computed every 500 trees.

Tune the model. This may take a few minutes.

```{r eval = TRUE}
xgb_tune_res <- tune_grid(
  xgb_tune_wf,
  resamples = cv_folds,
  grid = xgb_grid,
  metrics = class_metrics
)
```

```{r}
# you might want to cache and reload your tuning results...
# save(xgb_tune_res, file = './xgb_tune_res.RData')
# load(file = './xgb_tune_res.RData')
```

```{r eval = FALSE, echo = FALSE}
load(file = './xgb_tune_res.RData')
```



Extract the metrics computed using 10-fold CV:

```{r}
xgb_tune_metrics <- 
  xgb_tune_res |>
  collect_metrics()
xgb_tune_metrics
```

Here is a plot showing the misclassification error (lower values are better):

```{r}
xgb_tune_metrics |> 
  filter(.metric == "accuracy") |> 
  ggplot(aes(x = trees, y = 1 - mean, 
             colour = factor(tree_depth))) +
  geom_path() +
  labs(y = "Misclassification rate") + 
  scale_colour_manual(values = c("#D55E00", "#0072B2", "#009E73")) +
  facet_wrap(~ learn_rate, labeller = label_both) +
  labs(colour = 'tree_depth') + 
  theme_bw() +
  theme(legend.position = c(.98, .98), 
        legend.justification = c(1,1),
        legend.background = element_rect(colour = 'black'))
```

It seems that a lower learning rate (0.01) together with a tree depth of `2` or `3` and a smaller number of trees gives the best performance in terms of misclassification rate.

The goal, however, is to maximize sensitivity while maintaining high specificity. 

Look at sensitivity first:

```{r}
xgb_tune_metrics |> 
  filter(.metric == "sensitivity") |> 
  ggplot(aes(x = trees, y = mean, 
             colour = factor(tree_depth))) +
  geom_path() +
  labs(y = "Sensitivity") + 
  scale_colour_manual(values = c("#D55E00", "#0072B2", "#009E73")) +
  facet_wrap(~ learn_rate) +
  labs(colour = 'tree_depth') + 
  theme_bw() +
  theme(legend.position = c(.98, .98), 
        legend.justification = c(1,1),
        legend.background = element_rect(colour = 'black'))
```

We see that the smaller learning rate performs better when smaller trees (lower `tree_depth`) and fewer of them (lower `trees`) are used. 

Now look at specificity:

```{r}
xgb_tune_metrics |> 
  filter(.metric == "specificity") |> 
  ggplot(aes(x = trees, y = mean, 
             colour = factor(tree_depth))) +
  geom_path() +
  labs(y = "Specificity") + 
  scale_colour_manual(values = c("#D55E00", "#0072B2", "#009E73")) +
  facet_wrap(~ learn_rate) +
  labs(colour = 'tree_depth') + 
  theme_bw() +
  theme(legend.position = c(.98, .02), 
        legend.justification = c(1,0),
        legend.background = element_rect(colour = 'black'))
```

It looks like it will be hard to attain the goal of 70% sensitivity and 60% specificity using the model and data.

Here is a plot showing sensitivity and specificity together:

```{r}
xgb_tune_res |> 
  collect_metrics() |>
  filter(.metric %in% c("sensitivity", "specificity")) |>
  ggplot(aes(x = trees, y = mean, colour = .metric)) +
  geom_path() +
  facet_grid(learn_rate ~ tree_depth, labeller = label_both) +
  scale_colour_manual(values = c("#D55E00", "#0072B2")) +
  theme_bw() +
  labs(y = NULL) +
  theme(legend.position = c(.98, .2), 
        legend.justification = c(1,0),
        legend.background = element_rect(colour = 'black'))
```

We must now decide what values of the tuning parameters to use. If high sensitivity is the main goal, then we should consider a subset of models with a small learning rate and fewer trees. 

These are the results for `learn_rate < 0.1`, `tree_depth <= 2`, and `trees <= 6000`:

```{r}
xgb_tune_metrics |> 
  filter(learn_rate < 0.1 & tree_depth <= 2 & trees <= 6000) |> 
  select(trees:learn_rate, .metric, mean) |>
    pivot_wider(id_cols = trees:learn_rate,
                names_from = .metric,
                values_from = mean) |> 
  ggplot() +
  aes(x = specificity, y = sensitivity, 
      colour = factor(trees, ordered = TRUE), 
      size = learn_rate) +
  geom_point() +
  facet_wrap(~tree_depth, ncol = 1, labeller = label_both) +
  scale_size_continuous(range = c(2,4), breaks = 10^c(-3,-2)) +
  scale_colour_viridis_d(begin = .3, end = .9, option = 'E') +
  theme_bw() +
  labs(colour = 'trees')
```

The models with tree depth of 1 attain high sensitivity with few trees, with good specificity. When the learning rate is .01, there is a good tradeoff between sensitivity and specificity among models with tree depth 1, and the results appear somewhat insensitive to the number of trees used. Hence, consider only the models with tree depth of 1 and learning rate .01, and focus on the uncertainty in sensitivity estimates:

```{r}
xgb_tune_metrics |> 
  filter(learn_rate == 0.01 & tree_depth == 1 & trees <= 5000) |> 
  select(trees:learn_rate, .metric, mean, std_err) |> 
  filter(.metric %in% c("sensitivity", "specificity")) |> 
  mutate(low = mean - std_err, high = mean + std_err) |> 
  select(-std_err) |> 
    pivot_wider(id_cols = trees:learn_rate,
                names_from = .metric,
                values_from = c(mean, low, high)) |> 
  select(trees, specificity = mean_specificity, ends_with('sensitivity')) |> 
  ggplot() +
  aes(x = specificity,
      y = mean_sensitivity, ymin = low_sensitivity, ymax = high_sensitivity, 
      colour = factor(trees, ordered = TRUE)) +
  geom_pointrange() +
  geom_text(aes(label = trees), position = position_nudge(y = .01)) +
  scale_colour_viridis_d(begin = .3, end = .95, option = 'E') +
  theme_bw() +
  labs(colour = 'trees')
```

Among models that achieve 70% sensitivity within 1 SE, the model with 2500 trees has the highest specificity. 

```{r}
xgb_best <- 
  xgb_tune_metrics |> 
  filter(tree_depth == 1, learn_rate == 0.01, trees == 2500) |> 
  distinct(trees, tree_depth, learn_rate)
xgb_final_wf <- 
  finalize_workflow(xgb_tune_wf, xgb_best)
xgb_final_wf
```


# Test set performance

Train the finalized workflow on the entire training set and predict the test set:

```{r}
xgb_final_fit <- 
  xgb_final_wf |>
  last_fit(home_improv_split, metrics = class_metrics)
```

The results on the test set for class predictions are:

```{r}
xgb_test_results <- 
  xgb_final_fit |>
  collect_metrics()
xgb_test_results
```

How does this compare to the previous models? Below is a table summarizing the results from this tutorial, and the tutorials on decision trees and random forests (the exact values may differ when the notebooks are rerun):

Method              |  Accuracy   | Sensitivity  | Specificity    
--------------------|-------------|--------------|----------------
Classification tree | 0.646       | 0.629        | 0.648
Random Forest       | 0.620       | 0.660        | 0.614  
Boosting            | 0.614       | 0.670        | 0.606

We see that boosting improves on random forests in terms of sensitivity, but at the cost of lower specificity. In general, the more complex methods (i.e. forest and boosting) improve on sensitivity when compared to a single decision tree. 


# R session information

```{r}
sessionInfo()
```
