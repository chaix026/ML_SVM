# Support Vector Machine (SVM)
### What is Support Vector Machine (SVM)?

Support Vector Machine (SVM) is a supervised machine learning algorithm that can be used for both classification and regression tasks. However, it is primarily used for classification problems. SVM works by finding the hyperplane that best divides a dataset into classes. The best hyperplane is the one that maximizes the margin between the two classes. The points that lie on the boundary of the margin are called support vectors.

### Key Concepts of SVM:

1. **Hyperplane**: A decision boundary that separates different classes in the feature space.
2. **Margin**: The distance between the hyperplane and the nearest data points from either class. SVM aims to maximize this margin.
3. **Support Vectors**: Data points that are closest to the hyperplane and influence its position and orientation.

### Example of SVM in Python

We'll use the popular `scikit-learn` library to demonstrate SVM on a simple dataset.

#### Step-by-Step Example:

1. **Install scikit-learn**:

   If you haven't installed it yet, you can install `scikit-learn` using pip:

   ```bash
   pip install scikit-learn
   ```

2. **Import Libraries**:

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn import datasets
   from sklearn.model_selection import train_test_split
   from sklearn.svm import SVC
   from sklearn.metrics import accuracy_score
   ```

3. **Load Dataset**:

   We'll use the Iris dataset for this example.

   ```python
   # Load the Iris dataset
   iris = datasets.load_iris()
   X = iris.data[:, :2]  # Use only the first two features for visualization purposes
   y = iris.target

   # Split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   ```

4. **Train the SVM Model**:

   ```python
   # Create an instance of SVM and fit the data
   model = SVC(kernel='linear', C=1.0, random_state=42)
   model.fit(X_train, y_train)
   ```

5. **Make Predictions**:

   ```python
   # Make predictions on the test set
   y_pred = model.predict(X_test)

   # Evaluate the model
   accuracy = accuracy_score(y_test, y_pred)
   print(f'Accuracy: {accuracy:.2f}')
   ```

6. **Visualize the Decision Boundary**:

   ```python
   # Create a mesh to plot the decision boundary
   x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))

   # Plot the decision boundary
   Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
   Z = Z.reshape(xx.shape)
   plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

   # Plot the training points
   plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
   plt.xlabel('Sepal length')
   plt.ylabel('Sepal width')
   plt.title('SVM Decision Boundary')
   plt.show()
   ```

### Complete Code:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only the first two features for visualization purposes
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an instance of SVM and fit the data
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Create a mesh to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Plot the decision boundary
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM Decision Boundary')
plt.show()
```

This code demonstrates how to load a dataset, train an SVM model, evaluate its accuracy, and visualize the decision boundary. The `SVC` class from `sklearn.svm` is used to create the SVM model, and the `contourf` function from `matplotlib` is used to plot the decision boundary. Adjust the features and parameters as needed for your specific use case.
