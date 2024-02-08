# K-Nearest-Neighbour

### K-Nearest Neighbors (KNN) Algorithm:

The KNN algorithm is a simple and effective method for classification and regression tasks. Here's a brief overview along with formulas:

- **Algorithm**: Given a dataset with labeled points, KNN classifies new points based on the majority class of their k nearest neighbors.

- **Formula for Classification**:

  - For a new data point $\ x_q \$, the predicted class label $\ y_q \$, is determined by the majority class among its k nearest neighbors.

  $$\ y_q = \text{argmax} \left( \sum_{i=1}^{k} I(y_i = c) \right) \$$

  where $\ y_i \$, is the class label of the $\ i $th nearest neighbor, $\ c \$, represents the classes, and $\ I \$, is the indicator function.

- **Distance Metric**: Euclidean distance is commonly used to measure the similarity between data points in KNN.

  $$\ d(x_i, x_q) = \sqrt{\sum_{j=1}^{n} (x_{ij} - x_{qj})^2} \$$

  where $\ x_i \$, and $\ x_q \$, are data points, $\ x_{ij} \$, and $\ x_{qj} \$, are the $\ j \$th features of $\ x_i \$, and $\ x_q \$, respectively, and $\ n \$ is the number of features.

- **Choosing k**: The choice of k influences the model's performance and can be determined using techniques like cross-validation or grid search.

KNN is a non-parametric and lazy learning algorithm, meaning it doesn't make any assumptions about the underlying data distribution and doesn't require a training phase. It simply memorizes the training dataset and performs classification at runtime.




### Overview of Code Implementation:

#### Data Loading and Preprocessing:

- The code starts by loading data from a CSV file and performs preprocessing steps such as one-hot encoding for categorical variables and dropping unnecessary columns.

#### Train-Test Split:

- Splits the preprocessed data into training and testing sets using the `train_test_split` function from `sklearn.model_selection`.

#### Feature Scaling:

- Performs feature scaling using `StandardScaler` to standardize the feature variables.

#### K-Nearest Neighbors (KNN) Algorithm:

- Implements the K-Nearest Neighbors (KNN) algorithm to classify the data.
- Defines a function `KNearestNeighbours` that predicts the class labels of the test data based on the majority class of its k nearest neighbors in the training data.

#### Performance Evaluation:

- Evaluates the performance of the KNN classifier for different values of k by calculating accuracy.
- Selects the best value of k based on the highest accuracy achieved.

#### Making Predictions and Visualization:

- Makes predictions on the test data using the best value of k.
- Constructs a confusion matrix to visualize the classification results.


### Choosing k:
The choice of k significantly affects the performance of the KNN algorithm. A small k value may result in a noisy decision boundary, leading to overfitting, while a large k value may result in a smoother decision boundary, potentially missing important patterns in the data.

### Pros and Cons:

**Pros**:
- Simple and easy to implement.
- No training phase, making it computationally efficient.
- Non-parametric nature allows it to capture complex decision boundaries.

**Cons**:
- Computationally expensive during the prediction phase, especially for large datasets.
- Sensitive to the choice of distance metric and k value.
- Requires a sufficient amount of training data to produce accurate predictions.

#### Applications:

KNN is widely used in various domains, including:

- Pattern recognition
- Recommender systems
- Medical diagnosis
- Handwriting recognition
- Intrusion detection in cybersecurity

Understanding the KNN algorithm and its parameters is essential for effectively applying it to real-world problems and interpreting its results.
