library(data.table)
library(ggplot2)
library(mlr3)
library(mlr3learners)
library(data.table)
library(DoubleML)
library(checkmate)
library(Matrix)
lgr::get_logger("mlr3")$set_threshold("warn")
options(repr.plot.width=5, repr.plot.height=4)

library(paradox)
library(mlr3tuning)
library(mlr3pipelines)
library(mvtnorm)
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")





set.seed(1234)
n_rep = 200
nobs = 200
dimx = 400
#n_vars = 5
alpha_d = 0.5 # coeff of treatment variable

make_plr_DENSE = function(n_obs = 100, dim_x = 150, alpha = 1,
                                return_type = "DoubleMLData") {
  
  #assert_choice(
  #  return_type,
  #  c("data.table", "matrix", "data.frame", "DoubleMLData")
  #)
  # assert_count(n_obs)
  # assert_count(dim_x)
  # assert_numeric(alpha, len = 1)
  
  #a_0 = 1
  #a_1 = 0.25
  
  #a_0 = as.matrix(sqrt(3/2)* c( rep(1/sqrt(dim_x),floor(dim_x*2/3)), rep(0,dim_x-floor(dim_x*2/3)))) # a column
  #b_0 = as.matrix(sqrt(3/2)* c( rep(0,floor(dim_x*1/5)), rep(1/sqrt(dim_x),dim_x-floor(dim_x*1/5))))
  #a_0 = as.matrix(c( rep(1/sqrt(dim_x),floor(dim_x-3)), rep(1/3,3))) # a column
  #b_0 = as.matrix(c( rep(1/3,3), rep(1/sqrt(dim_x),dim_x-3)))
  
  # （1 1/4 1/9 ，。。。）
  a_0 = as.matrix(rep(1/sqrt(dim_x),dim_x)) # a column
  b_0 = as.matrix(rep(1/sqrt(dim_x),dim_x))
  
  # a_0 = matrix(1/(1:dim_x)^2, ncol = 1, byrow = TRUE)
  # b_0 = matrix(1/(1:dim_x)^2, ncol = 1, byrow = TRUE)
  s_2 = 1
  #cov_mat = toeplitz(0.5^(0:(dim_x - 1)))
  #x = rmvnorm(n = n_obs, mean = rep(0, dim_x), sigma = cov_mat)
  x = matrix(rnorm(n_obs * dim_x),nrow=n_obs)
  d = x %*% a_0 +rnorm(n_obs)  # treatment variable
  y = as.matrix(alpha_d * d +  x %*% b_0 + s_2 * rnorm(n_obs) )
    
  colnames(x) = paste0("X", 1:dim_x)
  colnames(y) = "y"
  colnames(d) = "d"
  if (return_type == "matrix") {
    return(list("X" = x, "y" = y, "d" = d))
  } else if (return_type == "data.frame") {
    data = data.frame(x, y, d)
    return(data)
  } else if (return_type == "data.table") {
    data = data.table(x, y, d)
    return(data)
  } else if (return_type == "DoubleMLData") {
    dt = data.table(x, y, d)
    data = DoubleMLData$new(dt, y_col = "y", d_cols = "d")
    return(data)
  }
}



data = list()
for (i_rep in seq_len(n_rep)) {
  set.seed(i_rep+6)
  data[[i_rep]] = make_plr_DENSE(n_obs=nobs, dim_x=dimx,
                                       return_type="data.frame")
}


# non_orth_score = function(y, d, l_hat, m_hat, g_hat, smpls) {
#   u_hat = y - g_hat
#   psi_a = -1*d*d
#   psi_b = d*u_hat
#   psis = list(psi_a = psi_a, psi_b = psi_b)
#   return(psis)
# }
# 
# set.seed(1111)




#ml_l = lrn("regr.cv_glmnet", alpha= 1) # alpha =1 is lasso
#ml_m = lrn("regr.cv_glmnet", alpha=1)
#ml_m = lrn("regr.xgboost", nrounds = 300, eta = 0.1) 
#ml_g = ml_l$clone()
# 
# theta_nonorth = rep(NA, n_rep)
# se_nonorth = rep(NA, n_rep)
# 
# for (i_rep in seq_len(n_rep)) {
#   cat(sprintf("Replication %d/%d", i_rep, n_rep), "\r", sep="")
#   flush.console()
#   df = data[[i_rep]]
#   obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
#   obj_dml_plr_nonorth = DoubleMLPLR$new(obj_dml_data,
#                                         ml_l, ml_m, ml_g,
#                                         n_folds=2,
#                                         score=non_orth_score,
#                                         apply_cross_fitting=FALSE)
#   obj_dml_plr_nonorth$fit()
#   
#   
#   theta_nonorth[i_rep] = obj_dml_plr_nonorth$coef
#   se_nonorth[i_rep] = obj_dml_plr_nonorth$se
# }
# 
# g_nonorth = ggplot(data.frame(theta_rescaled=(theta_nonorth - alpha)/se_nonorth)) +
#   geom_histogram(aes(y=after_stat(density), x=theta_rescaled, colour = "Non-orthogonal ML", fill="Non-orthogonal ML"),
#                  bins = 30, alpha = 0.3) +
#   geom_vline(aes(xintercept = 0), col = "black") +
#   suppressWarnings(geom_function(fun = dnorm, aes(colour = "N(0, 1)", fill="N(0, 1)"))) +
#   scale_color_manual(name='',
#                      breaks=c("Non-orthogonal ML", "N(0, 1)"),
#                      values=c("Non-orthogonal ML"="dark blue", "N(0, 1)"='black')) +
#   scale_fill_manual(name='',
#                     breaks=c("Non-orthogonal ML", "N(0, 1)"),
#                     values=c("Non-orthogonal ML"="dark blue", "N(0, 1)"=NA)) +
#   xlim(c(-6.0, 6.0)) + xlab("") + ylab("") + theme_minimal()
# 
# g_nonorth


# Define learner in a pipeline
# https://docs.doubleml.org/stable/guide/learners.html


## NEURAL NETWORK or OLS or OTHER METHODS
# set.seed(3141)
# ml_l = lrn("regr.nnet")
# ml_m = lrn("regr.nnet")
# obj_dml_data = DoubleMLData$new(data[[1]], y_col="y", d_cols="d")
# dml_plr_obj = DoubleMLPLR$new(obj_dml_data, ml_l, ml_m,n_folds=5,score='IV-type')
# dml_plr_obj$fit()
# dml_plr_obj$summary()
# 
# 
# ml_l = lrn("regr.xgboost")
# dml_plr_obj = DoubleMLPLR$new(obj_dml_data, ml_l , ml_m)
# dml_plr_obj$set_ml_nuisance_params("ml_l","d")
# dml_plr_obj$fit()
# dml_plr_obj$summary()




glmnet_pipe = po("learner",
                learner = lrn("regr.glmnet",alpha=0)) # alpha=1 is lasso

ml_m = as_learner(glmnet_pipe)
ml_l = as_learner(glmnet_pipe)
ml_g = as_learner(glmnet_pipe)

# double debiased ml
set.seed(6666)

theta_dml = rep(NA, n_rep)
se_dml = rep(NA, n_rep)

for (i_rep in seq_len(n_rep)) {
  cat(sprintf("Replication %d/%d", i_rep, n_rep), "\r", sep="")
  sprintf("Replication %d/%d", i_rep, n_rep)
  #flush.console()
  df = data[[i_rep]]
  dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")

  # Instantiate a DoubleML object
  dml_plr_obj = DoubleMLPLR$new(dml_data,ml_l, ml_g, ml_m,n_folds=5,score='IV-type') #'IV-type'
  #dml_plr_obj = DoubleMLPLR$new(dml_data,ml_l, ml_m,n_folds=5,score='partialling out')   
  # Parameter grid for lambda
  par_grids = ps(regr.glmnet.lambda = p_dbl(lower = 1e-3, upper = 0.1))
  
  tune_settings = list(terminator = trm("evals", n_evals = 200),
                       algorithm = tnr("grid_search", resolution = 50),
                       rsmp_tune = rsmp("cv", folds = 3),
                       measure = list("ml_l" = msr("regr.mse"),
                                      "ml_m" = msr("regr.mse")))
  dml_plr_obj$tune(param_set = list("ml_l" = par_grids,"ml_m" = par_grids,"ml_g" = par_grids),
                   tune_settings=tune_settings,
                   tune_on_fold=FALSE)
                   #tune_on_fold=TRUE)
  # THE FOLLOWING: score='partialling out'
  # dml_plr_obj$tune(param_set = list("ml_l" = par_grids,"ml_m" = par_grids),
  #                  tune_settings=tune_settings,
  #                  tune_on_fold=TRUE)

  dml_plr_obj$fit()
  dml_plr_obj$summary()
  theta_dml[i_rep] = dml_plr_obj$coef
  se_dml[i_rep] = dml_plr_obj$se
  #ttt = dml_plr_obj$.__enclos_env__$private$nuisance_est(dml_plr_obj$smpls[[1]])
}

# alpha in the next 3 line is only for the transparent of plot
g_dml = ggplot(data.frame(theta_rescaled=(theta_dml - alpha_d)/se_dml), aes(x = theta_rescaled)) +
  geom_histogram(aes(y=after_stat(density), x=theta_rescaled, colour = "Double ML with cross-fitting", fill="Double ML with cross-fitting"),
                 bins = 20, alpha = 0.3) +
  geom_vline(aes(xintercept = 0), col = "black") +
  suppressWarnings(geom_function(fun = dnorm, aes(colour = "N(0, 1)", fill="N(0, 1)"))) +
  scale_color_manual(name='',
                     breaks=c("Double ML with cross-fitting", "N(0, 1)"),
                     values=c("Double ML with cross-fitting"="dark green", "N(0, 1)"='black')) +
  scale_fill_manual(name='',
                    breaks=c("Double ML with cross-fitting", "N(0, 1)"),
                    values=c("Double ML with cross-fitting"="dark green", "N(0, 1)"=NA)) +
  xlim(c(-5.0, 5.0)) + xlab("") + ylab("") + theme_minimal()

g_dml
