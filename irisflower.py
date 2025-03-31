#  Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Define the feature column names
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']

# Load the Iris dataset from a file (assuming it's in the same directory as the script)
df = pd.read_csv('iris-flower-classification-project\iris.data', names=columns)

# Separate features (X) and target (Y)
data = df.values
X = data[:, 0:4]  # Features
Y = data[:, 4]    # Target (species)

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
logistic_regression = LogisticRegression(max_iter=1000)

# Train the Logistic Regression model on the training data
logistic_regression.fit(X_train, y_train)

y_pred = logistic_regression.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)