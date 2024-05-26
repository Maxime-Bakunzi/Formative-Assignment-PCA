# Principal Component Analysis (PCA) Project

## Overview
This project demonstrates the application of Principal Component Analysis (PCA) to reduce the dimensionality of a dataset while retaining as much information as possible. PCA is a key technique in data science for simplifying datasets, improving computational efficiency, and reducing noise.

## Project Steps

### 1. Standardize the Data
Before applying PCA, we standardize the data to ensure all features contribute equally. This involves transforming the data so that each feature has a mean of zero and a standard deviation of one.

### 2. Calculate the Covariance Matrix
The covariance matrix is calculated to understand the relationships between the features. Features with high covariance tend to vary together, indicating a potential reduction in dimensionality.

### 3. Perform Eigendecomposition
We perform eigendecomposition on the covariance matrix to obtain eigenvalues and eigenvectors. The eigenvalues indicate the amount of variance captured by each principal component, while the eigenvectors indicate the direction of these components.

### 4. Sort the Principal Components
Eigenvalues and eigenvectors are sorted in descending order of eigenvalues. This helps prioritize the principal components that capture the most variance, essential for effective dimensionality reduction.

### 5. Calculate Explained Variance
The explained variance for each principal component is calculated to understand how much information each component holds. This helps in selecting the most important components.

### 6. Reduce Data Dimension
Using the top principal components, the original data is transformed to a lower-dimensional space, preserving the most significant information.

## Benefits and Limitations of PCA

### Benefits
1. **Dimensionality Reduction**: Simplifies datasets by reducing the number of features.
2. **Noise Reduction**: Focuses on components that capture the most variance, reducing the impact of noise.

### Limitations
1. **Loss of Information**: May discard components that capture some variance, potentially losing valuable information.
2. **Interpretability**: The transformed features (principal components) can be difficult to interpret in the context of the original data.

## Code

Here is the complete code for the PCA process:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Original data
data = np.array([
    [   1,   2,  -1,   4,  10],
    [   3,  -3,  -3,  12, -15],
    [   2,   1,  -2,   4,   5],
    [   5,   1,  -5,  10,   5],
    [   2,   3,  -3,   5,  12],
    [   4,   0,  -3,  16,   2],
])

# Calculating the mean for each feature
mean = np.mean(data, axis = 0)
print("Mean of each feature:\n", mean)

# Calculating the standard deviation for each feature
std_dev = np.std(data, axis=0, ddof=0)
print("Standard Deviation of each feature:\n", std_dev)
# Standardizing the data
standardized_data = (data - mean)/ std_dev

print("Standardized Data:\n", standardized_data)

# Calculate the covariance matrix
cov_matrix = np.cov(standardized_data.T)
print("Covariance Matrix:\n", cov_matrix)

# Perform eigendecomposition on the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# np.argsort can only provide lowest to highest; use [::-1] to reverse the list

order_of_importance = np.argsort(eigenvalues)[:: -1]
print ( 'the order of importance is :\n {}'.format(order_of_importance))

# utilize the sort order to sort eigenvalues and eigenvectors
sorted_eigenvalues = eigenvalues[order_of_importance]

print('\n\n sorted eigen values:\n{}'.format(sorted_eigenvalues))
sorted_eigenvectors = eigenvectors[:, order_of_importance] # sort the columns
print('\n\n The sorted eigen vector matrix is: \n {}'.format(sorted_eigenvectors))

# Calculate the explained variance for each principal component
explained_variance = (sorted_eigenvalues / np.sum(sorted_eigenvalues)) * 100
explained_variance = ["{:.2f}%".format(value) for value in explained_variance]
print("Explained Variance:\n", explained_variance)

# Select the number of principal components
k = 2  # You can choose the number of principal components

# Reduce the data using the top k principal components
reduced_data = np.matmul(standardized_data, sorted_eigenvectors[:, :k])
print("Reduced Data:\n", reduced_data)
print("Shape of Reduced Data:\n", reduced_data.shape)
```
