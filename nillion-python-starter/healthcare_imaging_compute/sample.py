import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
health_data = pd.read_csv('data/cancer_data.csv')

# Replace values in the 'Diagnosis' column
health_data['Diagnosis'] = health_data['Diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Print the first 20 rows of data
print("First 20 rows of data' column:")
print(health_data.head(50))

# Remove ID column
health_data = health_data.drop('ID', axis=1)

# Split Dataset, 80 percent training, 20 percent testing
train_size = int(0.8 * len(health_data))
train_data = health_data[:train_size]
test_data = health_data[train_size:]

# Create a subset that is 25% of train_data
subset_size = int(0.25 * len(train_data))
subset_data = train_data[:subset_size]

###### Plotting the data ######

# Plot train data in one frame
# Calculate the number of rows and columns for subplots
num_plots = len(train_data.columns)
num_cols = 7  # Number of columns for subplots
num_rows = (num_plots - 1) // num_cols + 1  # Calculate the number of rows needed

# Used to modify desired height of the frame
height_per_row = 2  # Adjust this value as needed
fig_height = height_per_row * num_rows

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, fig_height))

# Plot distribution of each attribute
for i, column in enumerate(train_data.columns):
    row_index = i // num_cols
    col_index = i % num_cols
    sns.histplot(train_data[column], kde=True, ax=axes[row_index, col_index])
    axes[row_index, col_index].set_title(f'Distribution of {column}')
    axes[row_index, col_index].set_xlabel(column)
    axes[row_index, col_index].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

# Plot subset of train data in one frame
# Calculate the number of rows and columns for subplots
num_plots = len(subset_data.columns)
num_cols = 7  # Number of columns for subplots
num_rows = (num_plots - 1) // num_cols + 1  # Calculate the number of rows needed

# Used to modify desired height of the frame
height_per_row = 2  # Adjust this value as needed
fig_height = height_per_row * num_rows

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, fig_height))

# Plot distribution of each attribute
for i, column in enumerate(subset_data.columns):
    row_index = i // num_cols
    col_index = i % num_cols
    sns.histplot(subset_data[column], kde=True, ax=axes[row_index, col_index])
    axes[row_index, col_index].set_title(f'Distribution of {column}')
    axes[row_index, col_index].set_xlabel(column)
    axes[row_index, col_index].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

###### END Plotting the data ######

# Convert data to NumPy arrays
X_train = train_data.drop('Diagnosis', axis=1).values
X_train_subset = subset_data.drop('Diagnosis', axis=1).values
y_train = train_data['Diagnosis'].values
y_train_subset = subset_data['Diagnosis'].values


# Select a random test_data and drop 'Diagnosis' column
# random_row = test_data.sample(n=1)
random_row = test_data.sample(n=1, random_state=20) # fixed random_state=20, selects 541st row with Diagnosis=0
print("\nRandomly selected instance from test_data before removing target field:")
print(random_row)
X_test = random_row.drop('Diagnosis', axis=1).values

# Add a column of ones to X_train for the intercept term
X_train_augmented = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_train_subset_augmented = np.hstack((np.ones((X_train_subset.shape[0], 1)), X_train_subset))

# Calculate coefficients using the normal equation
theta = np.linalg.inv(X_train_augmented.T @ X_train_augmented) @ X_train_augmented.T @ y_train
theta_subset = np.linalg.inv(X_train_subset_augmented.T @ X_train_subset_augmented) @ X_train_subset_augmented.T @ y_train_subset

print("Coefficients (theta):")
print(theta)
print("Coefficients (theta_subset):")
print(theta_subset)

# Make predictions for the test data
X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
print("\nX_test_augmented:")
print(X_test_augmented)
prediction = X_test_augmented @ theta
prediction_subset = X_test_augmented @ theta_subset

print("Prediction for the test data:")
print(prediction)

print("\nPrediction for the test data using subset of train data:")
print(prediction_subset)

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