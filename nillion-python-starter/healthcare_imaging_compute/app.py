import numpy as np
import pandas as pd
from .utils.func import plot_distributions
import os

# Load dataset
script_directory = os.path.dirname(os.path.realpath(__file__))
csv_file_path = os.path.join(script_directory, 'data', 'cancer_data.csv')
health_data = pd.read_csv(csv_file_path)

# Replace values in the 'Diagnosis' column
health_data["Diagnosis"] = health_data["Diagnosis"].apply(
    lambda x: 1 if x == "M" else 0
)

# Print the first 20 rows of data
print("First 50 rows of data' column:")
print(health_data.head(50))

# Remove ID column
health_data = health_data.drop("ID", axis=1)

# Split Dataset, 80 percent training, 20 percent testing
train_size = int(0.8 * len(health_data))
train_data = health_data[:train_size]
test_data = health_data[train_size:]

# Create a subset that is 25% of train_data
subset_25_size = int(0.25 * len(train_data))
subset_75_size = int(0.75 * len(train_data))
subset_sm_data = train_data[:subset_25_size]

# Create a subset that is 75% of train_data
subset_lg_data = train_data[subset_25_size:]


# Plot datasets
plot_distributions(train_data, 7, 2, 25, 'Distribution Plots of Training Data')
plot_distributions(subset_sm_data, 7, 2, 25, 'Distribution Plots of Subset - Sm')
plot_distributions(subset_lg_data, 7, 2, 25, 'Distribution Plots of Subset - Lg')


# Set up data for training
X_train = train_data.drop("Diagnosis", axis=1).values
X_train_subset_sm = subset_sm_data.drop("Diagnosis", axis=1).values
X_train_subset_lg = subset_lg_data.drop("Diagnosis", axis=1).values
y_train = train_data["Diagnosis"].values
y_train_subset_sm = subset_sm_data["Diagnosis"].values
y_train_subset_lg = subset_lg_data["Diagnosis"].values

# Select a random test_data and drop 'Diagnosis' column
random_row = test_data.sample(n=1)
# random_row = test_data.sample(n=1, random_state=20) # fixed random_state=20, selects 541st row with Diagnosis=0
print("\nRandomly selected instance from test_data before removing target field:")
print(random_row)
X_test = random_row.drop("Diagnosis", axis=1).values

# Add a column of ones to X_train for the intercept term
X_train_augmented = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_train_subset_sm_augmented = np.hstack(
    (np.ones((X_train_subset_sm.shape[0], 1)), X_train_subset_sm)
)
X_train_subset_lg_augmented = np.hstack(
    (np.ones((X_train_subset_lg.shape[0], 1)), X_train_subset_lg)
)

# Calculate coefficients using the normal equation
theta = (
    np.linalg.inv(X_train_augmented.T @ X_train_augmented)
    @ X_train_augmented.T
    @ y_train
)
theta_subset_sm = (
    np.linalg.inv(X_train_subset_sm_augmented.T @ X_train_subset_sm_augmented)
    @ X_train_subset_sm_augmented.T
    @ y_train_subset_sm
)
theta_subset_lg = (
    np.linalg.inv(X_train_subset_lg_augmented.T @ X_train_subset_lg_augmented)
    @ X_train_subset_lg_augmented.T
    @ y_train_subset_lg
)

# print("Coefficients (theta):")
# print(theta)
# print("Coefficients (theta_subset_sm):")
# print(theta_subset_sm)
# print("Coefficients (theta_subset_sm):")
# print(theta_subset_lg)

# Make predictions for the test data
X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
# print("\nX_test_augmented:")
# print(X_test_augmented)
prediction = X_test_augmented @ theta
prediction_subset_sm = X_test_augmented @ theta_subset_sm
prediction_subset_lg = X_test_augmented @ theta_subset_lg

print("\nPrediction for the test data:")
print(prediction)

print("\nPrediction for the test data using the small subset of train data:")
print(prediction_subset_sm)
print("\nPrediction for the test data using the large subset of train data:")
print(prediction_subset_lg)

# Compute prediction value using combined thetas from subsets
# Calculate weights for each subset
weights_subset_sm = subset_25_size / train_size
weights_subset_lg = subset_75_size / train_size

# Use weighted average to combine thetas
combined_theta = (theta_subset_sm * weights_subset_sm) + (
    theta_subset_lg * weights_subset_lg
)
print("\nCombined theta:")
print(combined_theta)

prediction_subsets_combined = X_test_augmented @ combined_theta
print("\nPrediction for the test data using the combined theta:")
print(prediction_subsets_combined)

# Setup final computation without use of np
# Get the values from the augmented test data row
X_test_values = X_test_augmented[0]

# Initialize the prediction variable
prediction_wo_np = 0

# Perform element-wise multiplication and summation
for x_val, theta_val in zip(X_test_values, theta):
    prediction_wo_np += x_val * theta_val

print("Prediction for the test data without use of np:")
print(prediction_wo_np)

exit()