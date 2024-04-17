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

# Plot the distribution of the data set
plt_train_data = train_data.drop(['Diagnosis'], axis=1)

# Plot distribution of each attribute
for column in plt_train_data.columns:
    sns.histplot(plt_train_data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Convert data to NumPy arrays
X_train = train_data.drop('Diagnosis', axis=1).values
y_train = train_data['Diagnosis'].values

# Select a random test_data and drop 'Diagnosis' column
random_row = test_data.sample(n=1)
# random_row = test_data.sample(n=1, random_state=20) // fixed random_state
print("\nRandomly selected instance from test_data before removing target field:")
print(random_row)
X_test = random_row.drop('Diagnosis', axis=1).values

# Add a column of ones to X_train for the intercept term
X_train_augmented = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# Calculate coefficients using the normal equation
theta = np.linalg.inv(X_train_augmented.T @ X_train_augmented) @ X_train_augmented.T @ y_train

print("Coefficients (theta):")
print(theta)

# Make predictions for the test data
X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
prediction = X_test_augmented @ theta

print("Prediction for the test data:")
print(prediction)