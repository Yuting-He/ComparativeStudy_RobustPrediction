## Data preparation of the comparative study

## Objective:
## The goal is to construct two sets of low-dimensional datasets (7 dimensions and 15 dimensions) 
## using the datasets from the Ellenbach study and observe the performance of 
## different tuning methods on low-dimensional data.  

# Load the environment from scenariogrid_AUC.RData, which from Ellenbach's study
load("scenariogrid_AUC.RData")

## select 10 datasets with most observations
data_names <- paste0("data", 1:25)
data_rows <- sapply(data_names, function(x) nrow(get(x)))

data_info <- data.frame(name = data_names, rows = data_rows)

# select top 10 data sets
top_10_data <- data_info[order(-data_info$rows), ][1:10, ]

# rename data sets
for (i in 1:10) {
  assign(paste0("data_", i), get(top_10_data$name[i]))
}
# check name and observations
for (i in 1:10) {
  print(paste("Data:", paste0("data_", i), "Rows:", nrow(get(paste0("data_", i)))))
}


## feature selection
# create importance list
importance_list <- list()
reference_colnames <- NULL

for (i in 1:10) {
  data <- get(paste0("data_", i))
  
  # save column name of features
  if (is.null(reference_colnames)) {
    reference_colnames <- colnames(data)[-1]  # delete y
  } else {
    # check if column name is same as first 
    if (!all(colnames(data)[-1] == reference_colnames)) {
      stop(paste("colname unmatch! check the colname of data_", i, sep = ""))
    }
  }
  
  # random forest
  rf_model <- randomForest::randomForest(x = data[, -1],  # x
                           y = data[, 1],   # y
                           importance = TRUE)
  
  importance_list[[i]] <- randomForest::importance(rf_model, type = 1)  # type = 1: mean decrease of Gini
}

# check result
print(importance_list[[1]])

# importance matrix
importance_matrix <- do.call(cbind, importance_list)
# calculate mean value of each feature
importance_means <- rowMeans(importance_matrix)

# result
importance_df <- data.frame(
  Feature = reference_colnames,
  Importance = importance_means
)

importance_df <- importance_df[order(-importance_df$Importance), ]

print(importance_df)


################################################
# data selection with gap - inconstant gap, begin from 2nd
#  7 features
set.seed(123)
selected_indices <- c(2, 3, 4, 5, 10)
top_5_features <- importance_df$Feature[selected_indices]
remaining_features <- importance_df$Feature[-(1:10)]
random_features <- sample(remaining_features, 2)
selected_features <- c(top_5_features, random_features)

for (i in 1:10) {
  data <- get(paste0("data_", i))
  new_data <- data[, c("y", selected_features)]
  new_data_df <- as.data.frame(new_data)
  assign(paste0("dim7_gap_data_", i), new_data_df)
}


# 15 features
set.seed(123)
selected_indices <- c(2, 3, 4, 5, 7, 10, 15, 20, 30, 50)
top_10_features <- importance_df$Feature[selected_indices]
remaining_features <- importance_df$Feature[-(1:50)]
random_features <- sample(remaining_features, 5)
selected_features <- c(top_10_features, random_features)

for (i in 1:10) {
  data <- get(paste0("data_", i))
  new_data <- data[, c("y", selected_features)]
  new_data_df <- as.data.frame(new_data)
  assign(paste0("dim15_gap_data_", i), new_data_df)
}

# save data and scenariogrid
data_names <- c(paste0("dim7_gap_data_", 1:10), paste0("dim15_gap_data_", 1:10))

for (name in data_names) {
  data <- get(name)
  if ("matrix" %in% class(data)) {
    data <- as.data.frame(data)
    assign(name, data)
  }
}

# Add scenariogrid_lowdim to the list of objects to be saved
save(list = c(data_names, "scenariogrid_lowdim", "scenariogrid_lowdim_sample"), file = "compare_data_gap.Rda")