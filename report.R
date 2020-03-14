################################
# Setup
################################

knitr::opts_chunk$set(echo = TRUE)

# Install and attach the required add-on packages
required_packages <- c("dplyr", "caret", "corrplot", "GGally", "kernlab",
                       "C50", "klaR", "knitr", "kableExtra", "grid", "gridExtra")

for (package in required_packages) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package) 
  }
  library(package, character.only = TRUE)
}

# Load dataset
data <- read.csv("./biomechanical-features-of-orthopedic-patients/column_2C_weka.csv",
                 stringsAsFactors = FALSE)

# Remove temporary variables
rm(required_packages, package)

################################
# Analysis and data preparation
################################

glimpse(data)

# List of available classes
unique(data$class)

# Convert classes expressed as character strings to factors
data <- data %>% mutate(class = as.factor(class))

summary(data)

# Check for any not available variables
anyNA(data)

# Draw boxplot to highlight any possible outliers
# in the degree_spondylolisthesis variable distribution
boxplot(data$degree_spondylolisthesis,
        main="Degree of spondylolisthesis",
        cex.main = 0.8,
        col="gold")

# Reference: K. Ganguly, R Data Analysis Cookbook (2nd edition), Packt, 2017.
impute_outliers <- function(x, removeNA = TRUE){
    quantiles <- quantile(x, c(.05, .95), na.rm = removeNA)
    x[ x < quantiles[1] ] <- mean(x, na.rm = removeNA)
    x[ x > quantiles[2] ] <- median(x, na.rm = removeNA)
    x
}

# Apply mean/median imputation to degree_spondylolisthesis variable
data$degree_spondylolisthesis <- impute_outliers(data$degree_spondylolisthesis)

# Plot degree_spondylolisthesis variable distribution
# resulting from mean/median imputation
boxplot(data$degree_spondylolisthesis,
        main="Degree of spondylolisthesis \n with mean/median imputation",
        cex.main = 0.8,
        col="gold")

# Patients condition distribution: normal vs. abnormal
data %>% ggplot(aes(class, fill = class)) +
  geom_bar(stat = "count") +
  labs(x = "Patient Classification", y = "Number of patients") +
  ggtitle("Patients condition distribution: normal vs. abnormal") +
  theme_minimal()

# Plot variables correlation
M <- cor(data[,1:6])
corrplot(M, method = "number", tl.cex = 0.8, tl.col = "black")

# Plot the pelvic incidence distribution by patients' class
data %>% 
  ggpairs(columns = c(1, ncol(data)), aes(fill = class)) +
  ggtitle("Pelvic incidence distribution by patients' class") +
  theme_minimal()

# Split dataset in two subsets for training and validation
set.seed(100)
test_index <- createDataPartition(y = data$class, p = 0.2, list = FALSE)
training_set <- data[-test_index,]
validation_set <- data[test_index,]

# Remove temporary variables
rm(test_index)

################################
# Results
################################

# Check the computational outcome of each model via 10-fold cross validation
# with three repeats
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Model training via support vector machines with polynomial kernel
svm_model <- train(class~., data = training_set,
                   method = "svmPoly",
                   trControl= control,
                   tuneGrid = data.frame(degree = 1,
                                         scale = 1,
                                         C = 1),
                   preProcess = c("pca","scale","center")
)

# Predictions outcome
svm_predictions <- predict(svm_model, validation_set)

# Create confusion matrix
svm_confusion_matrix <- confusionMatrix(svm_predictions, validation_set$class)

# Store accuracy
accuracy_summary = tibble(Model = "Support vector machines with polynomial kernel",
                          Accuracy = svm_confusion_matrix$overall["Accuracy"])

# Print confusion matrix
svm_confusion_matrix

# Model training via C5.0 (decision tree)
decision_tree_model <- train(class~., data = training_set, 
                             method = "C5.0",
                             preProcess=c("scale", "center"),
                             trControl= control,
                             na.action = na.omit,
                             trace = FALSE
)

# Predictions outcome
decision_tree_predictions <- predict(decision_tree_model, validation_set)

# Create confusion matrix
C50_confusion_matrix <- confusionMatrix(decision_tree_predictions, validation_set$class)

# Store accuracy
accuracy_summary <- bind_rows(
  accuracy_summary,
  tibble(Model = "C5.0 (decision tree)",
         Accuracy = C50_confusion_matrix$overall["Accuracy"])
)

# Print confusion matrix
C50_confusion_matrix

# Naïve Bayes algorithm
naive_model <- train(class~., data = training_set,
                     method = "nb",
                     preProcess=c("scale","center"),
                     trControl= control
)

# Predictions outcome
naive_predictions <- predict(naive_model, validation_set, na.action = na.pass)

# Create confusion matrix
naive_bayes_confusion_matrix <- confusionMatrix(naive_predictions, validation_set$class)

accuracy_summary <- bind_rows(
  accuracy_summary,
  tibble(Model = "Naïve Bayes",
         Accuracy = naive_bayes_confusion_matrix$overall["Accuracy"])
)

naive_bayes_confusion_matrix

# Train model with neural network
neural_network_model <- train(class~., data = training_set,
                              method = "nnet",
                              trControl = control,
                              preProcess = c("scale","center"),
                              trace = FALSE
)

neural_network_predictions <- predict(neural_network_model, validation_set)

# Create confusion matrix
neural_network_confusion_matrix <- confusionMatrix(neural_network_predictions,
                                                   validation_set$class)

# Store accuracy
accuracy_summary <- bind_rows(
  accuracy_summary,
  tibble(Model = "Neural network",
         Accuracy = neural_network_confusion_matrix$overall["Accuracy"])
)

# Print confusion matrix
neural_network_confusion_matrix

# Compute the variables importance in each predictive model
svm_model_importance <- varImp(svm_model, scale = FALSE)
decision_tree_model_importance <- varImp(decision_tree_model, scale = FALSE)
naive_model_importance <- varImp(naive_model, scale = FALSE)
neural_network_importance <- varImp(neural_network_model, scale = FALSE)

# Plot the variables importance in each predictive model
p1 <- plot(svm_model_importance, main="Support vector machines \n with polynomial kernel ")
p2 <- plot(decision_tree_model_importance, main="C5.0 (decision tree)")
p3 <- plot(naive_model_importance, main="Naïve Bayes")
p4 <- plot(neural_network_importance, main="Neural network")

grid.arrange(p1, p2, p3, p4, ncol = 2)

################################
# Conclusion
################################

accuracy_summary %>%
  arrange(desc(Accuracy)) %>%
  knitr::kable() %>%
  kable_styling()

# Appendix: system configuration and R version

version

