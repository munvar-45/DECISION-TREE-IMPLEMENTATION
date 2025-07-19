# Import required libraries

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import pandas as pd

# Load the Iris dataset

iris = load_iris()

X = iris.data

y = iris.target

feature_names = iris.feature_names

target_names = iris.target_names

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Decision Tree Classifier

model = DecisionTreeClassifier(criterion='gini', random_state=42)

model.fit(X_train, y_train)

# Make predictions on test data

y_pred = model.predict(X_test)

# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Visualize the Decision Tree

plt.figure(figsize=(12, 8))

plot_tree(model, filled=True, feature_names=feature_names, class_names=target_names)

plt.title("Decision Tree Visualization - Iris Dataset")

plt.show()

# Optional: View Actual vs Predicted results

results_df = pd.DataFrame({

 "Actual": y_test,

 "Predicted": y_pred

})

print(results_df.head())
