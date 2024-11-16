
# Load the environment
load("compare_data_gap.Rda")

# Function: run_experiment
# Purpose: Evaluate classifiers and tuning methods using training and external datasets, and compute mean AUC.
# Inputs:
#   - dim: Dataset dimensionality (e.g., 7 or 15).
#   - index: Scenario index in the scenariogrid.
#   - scenariogrid: Data frame with Train, External dataset names, and Seed values.
# Process:
#   1. Load Train and External datasets based on the scenario.
#   2. Loop through tuning methods and classifiers.
#   3. Train models using `tuneandtrain`, validate on other datasets, and compute mean AUC.
#   4. Log best parameters and handle errors gracefully.
# Outputs:
#   - A list containing:
#     * results: Data frame with Method, Classifier, Train, External, Mean_AUC, and Best_Param.
#     * errors: List of error messages for debugging.

run_experiment <- function(dim, index, scenariogrid) {
  
  Train_name <- scenariogrid$Train[index]
  External_name <- scenariogrid$External[index]
  seed <- scenariogrid$Seed[index]
  
  # Dynamically get the train and external datasets
  Train <- get(paste0("dim", dim, "_", Train_name))
  External <- get(paste0("dim", dim, "_", External_name))
  
  # Define the results dataframe to store the output
  results_df <- data.frame(
    Method = character(),
    Classifier = character(),
    Train = character(),
    External = character(),
    Mean_AUC = numeric(),
    Best_Param = character(),  # New column for the best parameter
    stringsAsFactors = FALSE
  )
  
  # Create an empty list to store error information
  error_log <- list()
  
  # Set the seed for reproducibility
  set.seed(seed)
  
  # Define tuning methods and classifiers
  tuning_methods <- c("robusttunec", "ext", "int")
  classifiers <- c("boosting", "lasso", "ridge", "rf", "svm")
  
  # Perform experiments for each method and classifier
  for (method in tuning_methods) {
    for (classifier in classifiers) {
      # Try running the experiment, handle errors gracefully
      try_result <- try({
        # Print the method and classifier
        cat("Running with method:", method, "and classifier:", classifier, "\n")
        
        # Train the model using tuneandtrain
        model_result <- tuneandtrain(data = Train, dataext = External, tuningmethod = method, classifier = classifier)
        auc_results <- numeric()  # Initialize to empty numeric vector
        best_param <- NA  # Initialize best_param
        
        # Validation using the remaining 8 datasets
        for (i in setdiff(1:10, as.numeric(c(Train_name, External_name)))) {
          data <- get(paste0("dim", dim, "_data_", i))
          
          # Handle NA values
          if (anyNA(data)) {
            data <- na.omit(data)
          }
          
          # Choose prediction method based on classifier
          if (classifier %in% c("lasso", "ridge")) {
            predictions <- predict(model_result$best_model, newx = as.matrix(data[, -1]), s = model_result$best_lambda, type = "response")
            predicted_classes <- as.numeric(predictions)
            best_param <- model_result$best_lambda  # Store the best lambda
            
          } else if (classifier == "boosting") {
            predictions <- predict(model_result$best_model, newdata = as.matrix(data[, -1]), type = "response")
            predicted_classes <- as.numeric(predictions)
            best_param <- model_result$best_mstop  # Store the best mstop
            
          } else if (classifier == "rf") {
            pred_result <- predict(model_result$best_model, newdata = data)
            predicted_classes <- pred_result$data$prob.1
            best_param <- model_result$best_min_node_size  # Store the best min.node.size
            
          } else if (classifier == "svm") {
            data$y <- as.factor(data$y)
            pred_result <- predict(model_result$best_model, newdata = data)
            predicted_classes <- as.numeric(pred_result$data$response)
            best_param <- model_result$best_cost  # Store the best cost for SVM
            
          } else {
            stop("Unknown classifier")
          }
          
          # Check for length mismatch
          if (length(data[, 1]) == length(predicted_classes)) {
            auc_value <- pROC::auc(response = as.numeric(data[, 1]), predictor = predicted_classes)
            auc_results <- c(auc_results, auc_value)
          } else {
            stop(paste("Length mismatch in data:", paste0("dim", dim, "_data_", i)))
          }
        }
        
        # Calculate the mean AUC across the 8 validation datasets
        mean_auc <- if(length(auc_results) > 0) mean(auc_results, na.rm = TRUE) else NA
        
        # Add results to dataframe
        results_df <- rbind(results_df, data.frame(
          Method = method,
          Classifier = classifier,
          Train = Train_name,
          External = External_name,
          Mean_AUC = mean_auc,
          Best_Param = best_param,  # Store the best parameter
          stringsAsFactors = FALSE
        ))
        
      }, silent = TRUE)  # Use silent to suppress error messages
      
      # If there was an error, log NA for this combination and save the error message
      if (inherits(try_result, "try-error")) {
        error_message <- paste("Error in method:", method, "classifier:", classifier, "Train:", Train_name, "External:", External_name, "\nError:", try_result)
        cat(error_message, "\n")
        error_log[[length(error_log) + 1]] <- error_message  # Add error message to log
        
        # Add NA results to dataframe, ensure all columns have values
        results_df <- rbind(results_df, data.frame(
          Method = method,
          Classifier = classifier,
          Train = Train_name,
          External = External_name,
          Mean_AUC = NA,
          Best_Param = NA,  # Log NA for the parameter
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  
  # Return both the result dataframe and error log
  return(list(results = results_df, errors = error_log))
}



#######################################################
# Experiment in 7-dimensional

# run run_experiment()
library(RobustPrediction)
# 7 features - sample scenariogrid
dim <- "7_gap"
scenario <- scenariogrid_lowdim  
num_scenarios <- nrow(scenario)

# create result data frame
results_df_dim7_gap <- data.frame(
  Method = character(),
  Classifier = character(),
  Train = character(),
  External = character(),
  Mean_AUC = numeric(),
  Best_Param = character(),
  stringsAsFactors = FALSE
)

# create error list
errors_dim7_gap <- list()

# Set the save frequency 
save_frequency <- 5

for (index in 1:num_scenarios) {
  cat("Running experiment for dim 7, index:", index, "\n")
  
  experiment_result <- run_experiment(dim, index, scenariogrid = scenario)
  
  # save result into df
  results_df_dim7_gap <- rbind(results_df_dim7_gap, experiment_result$results)
  errors_dim7_gap <- c(errors_dim7_gap, experiment_result$errors)
  
  # print actual results_df_dim7
  cat("Current state of results_df_dim7_gap after iteration", index, ":\n")
  print(results_df_dim7_gap)
  
  # Save the latest result every `save_frequency` iterations
  if (index %% save_frequency == 0) {
    save(results_df_dim7_gap, errors_dim7_gap, file = "results_df_dim7_gap_latest.Rda")
  }
}

# print
print(results_df_dim7_gap)
if (length(errors_dim7_gap) > 0) {
  cat("\nErrors occurred during some experiments for dim 7:\n")
  print(errors_dim7_gap)
}

# save result
save(results_df_dim7_gap, file = "results_df_dim7_gap.Rda")


####################################
# Experiment in 15-dimensional
dim <- "15_gap"
scenario <- scenariogrid_lowdim  
num_scenarios <- nrow(scenario)

# create result data frame
results_df_dim15_gap <- data.frame(
  Method = character(),
  Classifier = character(),
  Train = character(),
  External = character(),
  Mean_AUC = numeric(),
  Best_Param = character(),
  stringsAsFactors = FALSE
)

# create error list
errors_dim15_gap <- list()

# Set the save frequency 
save_frequency <- 5

for (index in 1:num_scenarios) {
  cat("Running experiment for dim 15, index:", index, "\n")
  
  experiment_result <- run_experiment(dim, index, scenariogrid = scenario)
  
  # save result into df
  results_df_dim15_gap <- rbind(results_df_dim15_gap, experiment_result$results)
  errors_dim15_gap <- c(errors_dim15_gap, experiment_result$errors)
  
  # print actual results_df_dim15
  cat("Current state of results_df_dim15_gap after iteration", index, ":\n")
  print(results_df_dim15_gap)
  
  # Save the latest result every `save_frequency` iterations
  if (index %% save_frequency == 0) {
    save(results_df_dim15_gap, errors_dim15_gap, file = "results_df_dim15_gap_latest.Rda")
  }
}

# print
print(results_df_dim15_gap)
if (length(errors_dim15_gap) > 0) {
  cat("\nErrors occurred during some experiments for dim 7:\n")
  print(errors_dim15_gap)
}

# save result
save(results_df_dim15_gap, file = "results_df_dim15_gap.Rda")


#########################################
# Plot
# Scenario 1 with 7 features
# create facet boxplot
library(ggplot2)
boxplot_dim7_gap <- ggplot(results_df_dim7_gap, aes(x = Method, y = Mean_AUC)) +
  geom_boxplot() +
  facet_wrap(~Classifier, scales = "free") +  
  theme_bw() +  
  theme(strip.text.x = element_text(size = 12, face = "bold")) +  
  labs(title = "Prediction performance esstimates based on independent test data (7 features, feature selection with gap)",
       x = "Tuning Method",
       y = "Mean AUC") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

print(boxplot_dim7_gap)

results_df_dim7_gap <- results_df_dim7_gap7
results_df_dim15_gap <- results_df_dim7_gap7

# plot parameter
library(ggplot2)
param_dim7_gap <- ggplot(results_df_dim7_gap, aes(x = Method, y = Best_Param)) +
  geom_boxplot() +
  facet_wrap(~Classifier, scales = "free") +  
  theme_bw() +  
  theme(strip.text.x = element_text(size = 12, face = "bold")) +  
  labs(title = "Optimal values of tuning parameters (7 features, feature selection with gap)",
       x = "Tuning Method",
       y = "Optimal Hyperparameter") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

print(param_dim7_gap)

param_dim15_gap7 <- ggplot(results_df_dim15_gap7, aes(x = Method, y = Best_Param)) +
  geom_boxplot() +
  facet_wrap(~Classifier, scales = "free") +  
  theme_bw() +  
  theme(strip.text.x = element_text(size = 12, face = "bold")) +  
  labs(title = "Optimal values of tuning parameters (15 features, feature selection with gap)",
       x = "Tuning Method",
       y = "Optimal Hyperparameter") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

print(param_dim15_gap)


# Scenario 2 with 15 features
# create facet boxplot
library(ggplot2)
boxplot_dim15_gap <- ggplot(results_df_dim15_gap, aes(x = Method, y = Mean_AUC)) +
  geom_boxplot() +
  facet_wrap(~Classifier, scales = "free") +  
  theme_bw() +  
  theme(strip.text.x = element_text(size = 12, face = "bold")) +  
  labs(title = "Prediction performance esstimates based on independent test data (15 features, feature selection with gap)",
       x = "Tuning Method",
       y = "Mean AUC") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

print(boxplot_dim15_gap)


# importance boxplot
ggplot(importance_df, aes(y = Importance)) + 
  geom_boxplot() +
  labs(title = "Importance Boxplot", y = "Importance") +
  theme_minimal() +
  theme(axis.title.x = element_blank(), 
        axis.text.x = element_blank(),  
        axis.ticks.x = element_blank()) 
