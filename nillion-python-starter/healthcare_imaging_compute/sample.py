import numpy as np
import pandas as pd

# Load dataset
health_data = pd.read_csv('healthcare_imaging_compute/data/Sheet1.csv')

# Convert data to NumPy arrays
X = health_data.drop('Diagnosis', axis=1).values  # Assuming 'target_variable_name' is the name of your target variable
y = health_data['Diagnosis'].values

# Drop ID column
df_bank = health_data.drop('ID', axis=1)

# Assuming you have X_test data
X_test = ...

# Add a column of ones to X for the intercept term
X_augmented = np.hstack((np.ones((X.shape[0], 1)), X))

# Calculate coefficients using the normal equation
theta = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ y

print("Coefficients (theta):")
print(theta)

# Make predictions for the test data
X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
predictions = X_test_augmented @ theta

print("Predictions for the test data:")
print(predictions)
