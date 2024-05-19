
## Load Libraries

pacman::p_load(readr,janitor,vtable,tidyverse,tidymodels,DataExplorer,
               ggpubr,ggcorrplot,gghighlight,patchwork,doParallel,vip,
               PRROC,SHAPforxgboost,shapviz, explore, CalibratR, DataExplorer)

tidymodels_prefer()

all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

## Import data

df <- read_csv("heart_failure_clinical_records.csv")

df %>%  
  data.frame() %>% 
  clean_names -> df

## Explore data

st(df, out = "kable")

df %>% 
  plot_missing()

df %>% 
  plot_histogram()

df %>% 
  plot_correlation()

df %>% 
  mutate(death_event = factor(death_event)) %>% 
  ggplot(aes(x = time, fill = death_event)) +
  geom_histogram() +
  theme_pubr() +
  facet_grid(death_event~.)

df %>% 
  mutate(death_event = factor(death_event)) %>% 
  plot_boxplot(by = "death_event")


## Preprocessing

df %>% 
  mutate(death_event = factor(death_event)) -> df

set.seed(99)
df_split <- rsample::initial_split(
  df, 
  prop = 0.7, 
  strata = death_event
)

preprocessing_recipe <-
  recipes::recipe(death_event ~ ., data = training(df_split)) %>%
  recipes::step_nzv(all_nominal_predictors()) %>%     # Remove near zero variance
  step_YeoJohnson(all_numeric_predictors()) %>%       # Transform with Yeo-Johnson
  step_normalize(all_numeric_predictors()) %>%        # Normalize variables
  prep()

train <- recipes::bake(
  preprocessing_recipe, 
  new_data = training(df_split)
)

plot_histogram(train)

df_cv_folds <- 
  recipes::bake(
    preprocessing_recipe, 
    new_data = training(df_split)
  ) %>%  
  rsample::vfold_cv(v = 3)

## XGBoost model 1 specifications -------

xgboost_model <- 
  parsnip::boost_tree(
    trees = 1000,
    tree_depth = tune(), min_n = tune(),
    loss_reduction = 0,                     
    sample_size = tune(), mtry = 100,         
    learn_rate = tune(),   
    stop_iter = 10
  ) %>%
  set_engine("xgboost", nthread = 10) %>%
  set_mode("classification")

## Grid specification ---------

xgboost_params <-
  dials::parameters(
    tree_depth(),
    min_n(),
    #loss_reduction(),
    sample_size = sample_prop(),
    #finalize(mtry(), training(df_split)),
    learn_rate()
  )

xgboost_grid <-
  dials::grid_max_entropy(
    xgboost_params,
    size = 10
  )

## Define the workflow ----------

xgboost_wf <- 
  workflows::workflow() %>%
  add_model(xgboost_model) %>% 
  add_formula(death_event ~ .)

## Tune starting grid -----------

xgboost_ini <- tune::tune_grid(
  object = xgboost_wf,
  resamples = df_cv_folds,
  grid = xgboost_grid,
  control = tune::control_grid(verbose = T)
)

## Tune the model -----------

xgb_bo <-
  xgboost_wf %>%
  tune_bayes(
    resamples = df_cv_folds,
    metrics = metric_set(roc_auc),
    initial = xgboost_ini,
    param_info = xgboost_params,
    iter = 30,
    control = control_bayes(no_improve = 10,verbose = T)
  )

xgb_bo %>% 
  tune::show_best(metric = "roc_auc") %>%
  knitr::kable()

xgb_bo %>% 
  autoplot(type = "performance")

xgb_bo %>% 
  autoplot(type = "parameters")

xgb_bo %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  dplyr::select(mean, tree_depth:learn_rate) %>%
  pivot_longer(tree_depth:learn_rate,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC") 

xgboost_best_params <- xgb_bo %>%
  tune::select_best()

xgboost_best_params

xgboost_model %>% 
  finalize_model(xgboost_best_params) -> xgboost_model_final

final_xgb <- finalize_workflow(
  xgboost_wf,
  xgboost_best_params
)

final_xgb

final_xgb %>%
  fit(data = training(df_split)) %>%
  extract_fit_parsnip() %>%
  vip(geom = "col")

train_processed <- bake(preprocessing_recipe, new_data = training(df_split))

xgboost_model_fit <- xgboost_model_final %>%
  fit(
    formula = death_event ~ ., 
    data    = train_processed
  ) 

train_prediction <- xgboost_model_fit %>%
  predict_classprob.model_fit(new_data = train_processed) %>%
  bind_cols(training(df_split))

train_prediction %>% 
  dplyr::select(death_event,'1') %>% 
  dplyr::rename(death = death_event, pred = '1') -> train_prediction2

xgb.roc <- roc.curve(weights.class0 = as.numeric(as.character(train_prediction2$death)), scores.class0 = train_prediction2$pred, curve = T)

xgb.pr  <- pr.curve(weights.class0 = as.numeric(as.character(train_prediction2$death)), scores.class0 = train_prediction2$pred, curve = T)

xgb.realiability <- reliability_diagramm(as.numeric(as.character(train_prediction2$death)),train_prediction2$pred, 10)

plot(xgb.roc) 

plot(xgb.pr)

xgb.realiability

caret::confusionMatrix(train_prediction2$death
                       , factor(as.numeric(train_prediction2$pred > 0.5))
                       , positive = "1", mode = "everything")


## XGBoost model 2 specifications -------

## Without time attribute

df %>% 
  select(-time) -> df

set.seed(99)
df_split <- rsample::initial_split(
  df, 
  prop = 0.7, 
  strata = death_event
)

preprocessing_recipe <-
  recipes::recipe(death_event ~ ., data = training(df_split)) %>%
  recipes::step_nzv(all_nominal_predictors()) %>%     # Remove near zero variance
  step_YeoJohnson(all_numeric_predictors()) %>%       # Transform with Yeo-Johnson
  step_normalize(all_numeric_predictors()) %>%        # Normalize variables
  prep()

train <- recipes::bake(
  preprocessing_recipe, 
  new_data = training(df_split)
)

plot_histogram(train)

df_cv_folds <- 
  recipes::bake(
    preprocessing_recipe, 
    new_data = training(df_split)
  ) %>%  
  rsample::vfold_cv(v = 3)

xgboost_model <- 
  parsnip::boost_tree(
    trees = 1000,
    tree_depth = tune(), min_n = tune(),
    loss_reduction = 0,                     
    sample_size = tune(), mtry = 100,         
    learn_rate = tune(),   
    stop_iter = 10
  ) %>%
  set_engine("xgboost", nthread = 10) %>%
  set_mode("classification")

## Grid specification ---------

xgboost_params <-
  dials::parameters(
    tree_depth(),
    min_n(),
    #loss_reduction(),
    sample_size = sample_prop(),
    #finalize(mtry(), training(df_split)),
    learn_rate()
  )

xgboost_grid <-
  dials::grid_max_entropy(
    xgboost_params,
    size = 10
  )

## Define the workflow ----------

xgboost_wf <- 
  workflows::workflow() %>%
  add_model(xgboost_model) %>% 
  add_formula(death_event ~ .)

## Tune starting grid -----------

xgboost_ini <- tune::tune_grid(
  object = xgboost_wf,
  resamples = df_cv_folds,
  grid = xgboost_grid,
  control = tune::control_grid(verbose = T)
)

## Tune the model -----------

xgb_bo <-
  xgboost_wf %>%
  tune_bayes(
    resamples = df_cv_folds,
    metrics = metric_set(roc_auc),
    initial = xgboost_ini,
    param_info = xgboost_params,
    iter = 30,
    control = control_bayes(no_improve = 10,verbose = T)
  )

# xgb_bo %>% collect_metrics()

xgb_bo %>% 
  tune::show_best(metric = "roc_auc") %>%
  knitr::kable()

xgb_bo %>% 
  autoplot(type = "performance")

xgb_bo %>% 
  autoplot(type = "parameters")

xgb_bo %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  dplyr::select(mean, tree_depth:learn_rate) %>%
  pivot_longer(tree_depth:learn_rate,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC") 

xgboost_best_params <- xgb_bo %>%
  tune::select_best()

xgboost_best_params

xgboost_model %>% 
  finalize_model(xgboost_best_params) -> xgboost_model_final

final_xgb <- finalize_workflow(
  xgboost_wf,
  xgboost_best_params
)

final_xgb

final_xgb %>%
  fit(data = training(df_split)) %>%
  extract_fit_parsnip() %>%
  vip(geom = "col")

train_processed <- bake(preprocessing_recipe, new_data = training(df_split))

xgboost_model_fit <- xgboost_model_final %>%
  fit(
    formula = death_event ~ ., 
    data    = train_processed
  ) 

train_prediction <- xgboost_model_fit %>%
  predict_classprob.model_fit(new_data = train_processed) %>%
  bind_cols(training(df_split))

train_prediction %>% 
  dplyr::select(death_event,'1') %>% 
  dplyr::rename(death = death_event, pred = '1') -> train_prediction2

xgb.roc <- roc.curve(weights.class0 = as.numeric(as.character(train_prediction2$death)), scores.class0 = train_prediction2$pred, curve = T)

xgb.pr  <- pr.curve(weights.class0 = as.numeric(as.character(train_prediction2$death)), scores.class0 = train_prediction2$pred, curve = T)

xgb.realiability <- reliability_diagramm(as.numeric(as.character(train_prediction2$death)),train_prediction2$pred, 10)

plot(xgb.roc) 

plot(xgb.pr)

xgb.realiability

caret::confusionMatrix(train_prediction2$death
                       , factor(as.numeric(train_prediction2$pred > 0.5))
                       , positive = "1", mode = "everything")

## Logit model 3 specifications -------

logit_model <- logistic_reg(
  penalty = tune(),
  mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

## Grid specification ---------

logit_params <-
  dials::parameters(
    mixture(),
    penalty()
  )

xgboost_grid <-
  dials::grid_max_entropy(
    logit_params,
    size = 10
  )

## Define the workflow ----------

logit_wf <- 
  workflows::workflow() %>%
  add_model(logit_model) %>% 
  add_formula(death_event ~ .)

## Tune starting grid -----------

logit_ini <- tune::tune_grid(
  object = logit_wf,
  resamples = df_cv_folds,
  grid = xgboost_grid,
  control = tune::control_grid(verbose = T)
)

## Tune the model -----------

logit_bo <-
  logit_wf %>%
  tune_bayes(
    resamples = df_cv_folds,
    metrics = metric_set(roc_auc),
    initial = logit_ini,
    param_info = logit_params,
    iter = 30,
    control = control_bayes(no_improve = 10,verbose = T)
  )

# xgb_bo %>% collect_metrics()

logit_bo %>% 
  tune::show_best(metric = "roc_auc") %>%
  knitr::kable()


logit_bo %>% 
  autoplot(type = "performance")

logit_bo %>% 
  autoplot(type = "parameters")

logit_bo %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  dplyr::select(mean, mixture:penalty) %>%
  pivot_longer(mixture:penalty,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC") 

logit_best_params <- logit_bo %>%
  tune::select_best()

logit_best_params

logit_model %>% 
  finalize_model(logit_best_params) -> logit_model_final

final_logit <- finalize_workflow(
  logit_wf,
  logit_best_params
)

final_logit

final_logit %>%
  fit(data = training(df_split)) %>%
  extract_fit_parsnip() %>%
  vip(geom = "col")

train_processed <- bake(preprocessing_recipe, new_data = training(df_split))

logit_model_fit <- logit_model_final %>%
  fit(
    formula = death_event ~ ., 
    data    = train_processed
  ) 

train_prediction <- logit_model_fit %>%
  predict_classprob.model_fit(new_data = train_processed) %>%
  bind_cols(training(df_split))

train_prediction %>% 
  dplyr::select(death_event,'1') %>% 
  dplyr::rename(death = death_event, pred = '1') -> train_prediction2

xgb.roc <- roc.curve(weights.class0 = as.numeric(as.character(train_prediction2$death)), scores.class0 = train_prediction2$pred, curve = T)

xgb.pr  <- pr.curve(weights.class0 = as.numeric(as.character(train_prediction2$death)), scores.class0 = train_prediction2$pred, curve = T)

xgb.realiability <- reliability_diagramm(as.numeric(as.character(train_prediction2$death)),train_prediction2$pred, 10)

plot(xgb.roc) 

plot(xgb.pr)

xgb.realiability

caret::confusionMatrix(train_prediction2$death
                       , factor(as.numeric(train_prediction2$pred > 0.5))
                       , positive = "1", mode = "everything")

## Final model en test set

test_processed <- bake(preprocessing_recipe, new_data = testing(df_split))

test_prediction <- xgboost_model_fit %>%
  predict_classprob.model_fit(new_data = test_processed) %>%
  bind_cols(testing(df_split))

test_prediction %>% 
  dplyr::select(death_event,'1') %>% 
  dplyr::rename(death = death_event, pred = '1') -> test_prediction2

xgb.roc <- roc.curve(weights.class0 = as.numeric(as.character(test_prediction2$death)), scores.class0 = test_prediction2$pred, curve = T)

xgb.pr  <- pr.curve(weights.class0 = as.numeric(as.character(test_prediction2$death)), scores.class0 = test_prediction2$pred, curve = T)

xgb.realiability <- reliability_diagramm(as.numeric(as.character(test_prediction2$death)),test_prediction2$pred, 10)

plot(xgb.roc) 

plot(xgb.pr)

xgb.realiability

caret::confusionMatrix(test_prediction2$death
                       , factor(as.numeric(test_prediction2$pred > 0.5))
                       , positive = "1", mode = "everything")

## SHAP VAlues model explaination

library(shapviz)

df2  <- bake(preprocessing_recipe
             , new_data = df
             , has_role("predictor")
             , composition = "matrix")

shap <- shapviz(extract_fit_engine(xgboost_model_fit)
                , X_pred = df2
                , which_class = '1')

sv_importance(shap, kind = "both", show_numbers = TRUE, bee_width = 0.2)

sv_dependence(shap, "serum_creatinine", color_var = "ejection_fraction")

sv_force(shap, row_id = 1L)

sv_waterfall(shap, row_id = 1L)

sv_waterfall(shap, row_id = 5L)

shap$S %>% 
  data.frame() %>% 
  tibble() -> shap.df
shap.df$baseline <- shap$baseline

shap.df$total_shap <- rowSums(shap.df) 
shap.df$probability <- 1/(1+exp(-shap.df$total_shap))

shap.df %>% 
  head(10) %>% 
  select(total_shap, probability)

shap2 <- shap

df %>% 
  select(- death_event) -> shap2$X

sv_force(shap2, row_id = 1L) / sv_force(shap2, row_id = 5L)








































