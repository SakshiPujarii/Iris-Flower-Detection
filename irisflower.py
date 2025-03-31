# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Define the feature column names
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']

# Load the Iris dataset from UCI repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, names=columns)

# Separate features (X) and target (Y)
X = df.iloc[:, :-1].values  # Features
Y = df.iloc[:, -1].values   # Target (species)

# Encode target labels to numerical values
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)  # Convert species names to numbers (0,1,2)

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train a Logistic Regression model
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train, y_train)

# Predictions
y_pred = logistic_regression.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix & Heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Decision Boundary Visualization (Using 2 features for plotting)
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model):
    X_set, y_set = X[:, [0, 2]], y  # Select two features for plotting
    X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.1),
                         np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.1))
    
    plt.figure(figsize=(8,6))
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.3, cmap=ListedColormap(('red', 'blue', 'green')))
    
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'blue', 'green'))(i), label=label_encoder.classes_[i])
    
    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Length")
    plt.legend()
    plt.title("Decision Boundary for Iris Classification")
    plt.show()

# Plot decision boundary using only Sepal Length and Petal Length
plot_decision_boundary(X_train, y_train, logistic_regression)

# Predict a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example input
predicted_species = label_encoder.inverse_transform(logistic_regression.predict(new_sample))
print(f"\nPredicted species for input {new_sample[0]}: {predicted_species[0]}")
