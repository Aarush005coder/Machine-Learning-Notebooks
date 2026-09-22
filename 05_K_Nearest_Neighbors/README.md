# K-Nearest Neighbors (KNN)

A practical and beginner-friendly collection of **Jupyter Notebooks** covering **K-Nearest Neighbors (KNN)**, one of the fundamental supervised machine learning algorithms used for **classification and regression** problems.

This section focuses on understanding the core concepts of KNN, distance-based learning, feature scaling, choosing the optimal value of **K**, model training, predictions, decision boundaries, and evaluating KNN performance using standard classification and regression techniques.

---

## 📚 Contents

| **#**  | **Notebook**                                                                                                                                                               | **Description**                                                                                                                                                                                                                             |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **01** | [`K_Nearest_Neighbors_01_Basics.ipynb`](https://github.com/Aarush005coder/Machine-Learning-Notebooks/blob/main/05_K_Nearest_Neighbors/K_Nearest_Neighbors_01_Basics.ipynb) | Fundamentals of KNN, dataset exploration, feature and target selection, train-test split, feature scaling, KNN classification, predictions, model evaluation, K selection, distance metrics, decision boundaries, and basic KNN regression. |

---

## 🧠 What is K-Nearest Neighbors?

**K-Nearest Neighbors (KNN)** is a **supervised machine learning algorithm** used for both **classification and regression** problems.

KNN is a **non-parametric, instance-based learning algorithm** that makes predictions based on the similarity between data points.

The basic idea is:

> **Similar data points tend to have similar outcomes.**

Instead of learning an explicit mathematical equation during training, KNN stores the training data and uses the nearest observations to make predictions for a new data point.

For classification, KNN uses **majority voting** among the nearest neighbors.

For regression, KNN generally uses the **average or weighted average** of the nearest neighbors.

---

## 🔢 How KNN Works

For a new data point, KNN follows these basic steps:

```text
New Data Point
      ↓
Calculate Distance
      ↓
Find Nearest Data Points
      ↓
Select K Neighbors
      ↓
       ┌───────────────┐
       ↓               ↓
Classification     Regression
       ↓               ↓
Majority Vote      Average Value
       ↓               ↓
   Prediction       Prediction
```

The value of **K** determines how many neighboring observations are considered.

For example, if:

```text
K = 3
```

the algorithm considers the **3 nearest data points**.

---

## 📐 Distance Calculation

KNN relies heavily on distance calculations to determine which observations are closest to a new data point.

### Euclidean Distance

The most commonly used distance metric is **Euclidean Distance**.

For two points:

```text
P = (x₁, x₂)
Q = (y₁, y₂)
```

the Euclidean distance is:

```text
d(P,Q) = √((x₁-y₁)² + (x₂-y₂)²)
```

For `n` dimensions:

```text
d(P,Q) = √(Σ(xᵢ-yᵢ)²)
```

A smaller distance means that two observations are more similar according to the selected distance metric.

---

## 🏷️ KNN Classification

KNN classification is used when the target variable represents a **category or class**.

### Example

Suppose we have:

```text
🔵 🔵 🔵       🟢 🟢
      ?
```

If the unknown point is closer to the blue observations, KNN may classify it as:

```text
Prediction → 🔵 Blue Class
```

### Majority Voting

For:

```text
K = 5
```

suppose the nearest neighbors are:

```text
Neighbor 1 → Class A
Neighbor 2 → Class A
Neighbor 3 → Class B
Neighbor 4 → Class A
Neighbor 5 → Class B
```

The votes are:

```text
Class A → 3
Class B → 2
```

Therefore:

```text
Prediction → Class A
```

---

## 📊 Dataset Used

The basics notebook demonstrates KNN classification using a simple customer dataset.

### Features

```text
Age
Salary
```

### Target

```text
Purchased
```

where:

```text
0 → Not Purchased
1 → Purchased
```

### Objective

```text
Age + Salary
     ↓
    KNN
     ↓
Purchase Prediction
```

The model learns from existing customer observations and predicts whether a new customer is likely to purchase the product based on nearby observations.

---

## 📏 Feature Scaling

Feature scaling is particularly important for KNN because it is a **distance-based algorithm**.

Suppose the dataset contains:

```text
Age       → 20–60
Salary    → 20,000–100,000
```

Salary has much larger numerical values than Age.

Without scaling, Salary can have a much larger influence on the distance calculation.

Therefore, features are standardized before applying KNN.

### Standardization

```text
z = (x - μ) / σ
```

Where:

* **x** → Original value
* **μ** → Mean
* **σ** → Standard deviation
* **z** → Standardized value

In Scikit-learn:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

It is important to fit the scaler on the training data and use that fitted scaler to transform the test data.

---

## 🎯 Choosing the Value of K

The value of **K** is one of the most important hyperparameters in KNN.

For example:

```text
K = 1
K = 3
K = 5
K = 7
...
```

Different values of K can produce different predictions and decision boundaries.

The notebook evaluates multiple K values and compares their performance.

```text
K
↓
Train / Evaluate KNN
↓
Calculate Accuracy
↓
Compare Results
↓
Select K based on validation performance
```

---

## ⚖️ Small K vs Large K

The choice of K affects the model's bias and variance.

### Small K

Example:

```text
K = 1
```

Characteristics:

* Very sensitive to nearby observations
* Flexible decision boundary
* Can be sensitive to noise
* Can lead to overfitting

Conceptually:

```text
Low Bias
High Variance
```

### Large K

Characteristics:

* Smoother decision boundary
* Less sensitive to individual observations
* Can ignore local patterns
* Can lead to underfitting

Conceptually:

```text
High Bias
Low Variance
```

Therefore, K should be selected using an appropriate validation procedure rather than simply choosing the smallest or largest value.

---

## 📈 K vs Accuracy

The notebook evaluates different values of K and visualizes their performance.

Example:

```text
Accuracy
   ↑
   │       ●
   │    ●     ●
   │  ●         ●
   │
   └──────────────────→ K
      1  3  5  7  9
```

This visualization helps understand how model performance changes as K changes.

---

## 🔀 Train-Test Split

The dataset is divided into training and testing sets.

```text
Dataset
   │
   ├───────────────┐
   ↓               ↓
Training Data   Testing Data
   │               │
   ↓               ↓
Train KNN       Evaluate KNN
```

The **training set** contains observations used by the KNN model.

The **testing set** contains unseen observations used to evaluate the model.

In the notebook:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## 🤖 Model Training

KNN can be implemented using Scikit-learn's `KNeighborsClassifier`.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=3
)

knn.fit(X_train_scaled, y_train)
```

Unlike many parametric algorithms, KNN does not learn a set of coefficients or a fixed equation during training.

The training data is retained and used during prediction to find the nearest observations.

---

## 🔮 Prediction

After training the model, predictions can be generated using:

```python
y_pred = knn.predict(X_test_scaled)
```

For a new customer:

```python
new_customer = [[41, 62000]]

new_customer_scaled = scaler.transform(new_customer)

prediction = knn.predict(new_customer_scaled)

print(prediction)
```

The model compares the new observation with the stored training observations and uses the nearest neighbors to generate the prediction.

---

## 📊 Model Evaluation

The notebook uses several classification metrics to evaluate KNN performance.

### 1. Accuracy

Accuracy represents the proportion of correctly classified observations.

```text
Accuracy =
Correct Predictions / Total Predictions
```

In Scikit-learn:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
```

A higher accuracy indicates that a larger proportion of evaluated observations were classified correctly, but accuracy should be interpreted in the context of the dataset and class distribution.

---

### 2. Confusion Matrix

A confusion matrix summarizes classification results by comparing actual classes with predicted classes.

For binary classification:

```text
                    Predicted
                  0         1
Actual  0        TN        FP
        1        FN        TP
```

Where:

* **TP** → True Positive
* **TN** → True Negative
* **FP** → False Positive
* **FN** → False Negative

In the notebook:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
```

---

### 3. Precision

Precision measures how many observations predicted as positive were actually positive.

```text
Precision = TP / (TP + FP)
```

High precision means that the model produces relatively fewer false-positive predictions.

---

### 4. Recall

Recall measures how many actual positive observations were correctly identified.

```text
Recall = TP / (TP + FN)
```

High recall means that the model identifies a larger proportion of actual positive observations.

---

### 5. F1-Score

F1-score combines precision and recall using their harmonic mean.

```text
F1 = 2 × (Precision × Recall)
     / (Precision + Recall)
```

It is useful when both precision and recall are important.

---

## 📐 Distance Metrics

KNN can use different distance metrics.

### 1. Euclidean Distance

```text
d = √(Σ(xᵢ-yᵢ)²)
```

Commonly used for continuous numerical features.

### 2. Manhattan Distance

```text
d = Σ|xᵢ-yᵢ|
```

Also known as **L1 distance**.

### 3. Minkowski Distance

Minkowski distance generalizes several distance measures.

```text
d = (Σ|xᵢ-yᵢ|ᵖ)¹/ᵖ
```

For example:

```python
KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)
```

---

## 🗺️ Decision Boundary

A **decision boundary** represents regions of the feature space where the model predicts different classes.

Conceptually:

```text
Feature 2
   ↑
   │       Class B
   │   ● ● ● ●
   │
   │────────────── Decision Boundary
   │
   │   ● ● ●
   │       Class A
   └──────────────────→ Feature 1
```

KNN can produce non-linear and flexible decision boundaries because predictions depend on the local distribution of training observations.

The notebook visualizes the KNN decision boundary to demonstrate how changing K can affect the classification regions.

---

## 📉 KNN for Regression

KNN can also be used for **regression problems**.

Instead of selecting the most common class, KNN regression predicts a numerical value based on neighboring observations.

Scikit-learn provides:

```python
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(
    n_neighbors=2
)

knn_reg.fit(X_train, y_train)

predictions = knn_reg.predict(X_test)
```

### Classification vs Regression

| **Feature**            | **Classification**        | **Regression**             |
| ---------------------- | ------------------------- | -------------------------- |
| **Output**             | Class / Category          | Continuous Numerical Value |
| **Scikit-learn Class** | `KNeighborsClassifier`    | `KNeighborsRegressor`      |
| **Prediction Method**  | Majority Voting           | Average / Weighted Average |
| **Example**            | Purchased / Not Purchased | House Price                |

---

## ⚙️ Important KNN Parameters

### `n_neighbors`

Controls the number of neighbors considered.

```python
KNeighborsClassifier(
    n_neighbors=5
)
```

---

### `weights`

Controls how neighbors contribute to the prediction.

#### Uniform

```python
weights="uniform"
```

All neighbors have equal influence.

#### Distance

```python
weights="distance"
```

Closer neighbors receive greater influence.

---

### `metric`

Controls how distance is calculated.

Examples:

```python
metric="euclidean"
metric="manhattan"
metric="minkowski"
```

---

### `algorithm`

Scikit-learn can use different algorithms for finding neighbors, depending on the data and configuration.

```python
algorithm="auto"
```

Other options include:

```text
ball_tree
kd_tree
brute
```

---

## 📊 Visualizations

The notebook includes visualizations such as:

* **Feature Scatter Plot**
* **K vs Accuracy Plot**
* **Confusion Matrix**
* **KNN Decision Boundary**
* **Actual vs Predicted Values**
* **Classification Results**

These visualizations help understand the relationship between the observations, neighboring points, K selection, and model predictions.

---

## 🔬 Manual Distance Example

The notebook also demonstrates the basic idea of distance calculation manually.

For two points:

```text
P = (2, 3)
Q = (5, 7)
```

Euclidean distance:

```text
d = √((5-2)² + (7-3)²)

  = √(3² + 4²)

  = √25

  = 5
```

This simple calculation forms the basis of the nearest-neighbor search used by KNN.

---

## 🎯 KNN Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Feature & Target Selection
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Choose K
   ↓
Calculate Distances
   ↓
Find K Nearest Neighbors
   ↓
Majority Voting / Average
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Tune K
   ↓
Final Model
```

---

## 🎯 Learning Objectives

After completing this notebook, you should be able to:

* **Explain K-Nearest Neighbors**
* **Understand instance-based learning**
* **Understand non-parametric learning**
* **Explain how KNN makes predictions**
* **Understand the role of K**
* **Calculate Euclidean distance**
* **Understand different distance metrics**
* **Understand why feature scaling is important**
* **Implement KNN using Scikit-learn**
* **Perform KNN classification**
* **Perform basic KNN regression**
* **Make predictions on unseen data**
* **Evaluate classification performance**
* **Interpret a confusion matrix**
* **Understand precision, recall, and F1-score**
* **Analyze the effect of different K values**
* **Understand overfitting and underfitting**
* **Visualize KNN decision boundaries**
* **Understand important KNN hyperparameters**

---

## 🌍 Real-World Applications

KNN can be used for various machine learning problems, including:

* **Customer Classification**
* **Recommendation Systems**
* **Pattern Recognition**
* **Image Classification**
* **Handwritten Digit Recognition**
* **Medical Classification**
* **Customer Segmentation Support**
* **Anomaly Detection**
* **Similarity Search**
* **House Price Estimation**
* **Product Recommendation**
* **Document Classification**

The suitability of KNN depends on the dataset size, feature representation, dimensionality, distance metric, and computational requirements.

---

## ⚠️ Important Considerations

### 1. Feature Scaling

Because KNN relies on distances, features with different numerical scales can distort the distance calculation.

---

### 2. Choice of K

A very small K can make the model sensitive to noise, while a very large K can smooth away useful local patterns.

---

### 3. Computational Cost

KNN can require substantial computation during prediction because distances to training observations may need to be evaluated.

---

### 4. High Dimensionality

KNN can become less effective as the number of dimensions increases.

This is related to the **curse of dimensionality**.

---

### 5. Irrelevant Features

Irrelevant or noisy features can affect distance calculations and therefore influence predictions.

Feature selection and preprocessing can therefore be important.

---

## 🛠️ Technologies & Libraries

| **Category**                | **Technologies** |
| --------------------------- | ---------------- |
| **Programming Language**    | Python           |
| **Data Manipulation**       | NumPy, Pandas    |
| **Data Visualization**      | Matplotlib       |
| **Machine Learning**        | Scikit-learn     |
| **Development Environment** | Jupyter Notebook |

---

## 📦 Installation

Install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Or install all dependencies from the main repository:

```bash
pip install -r ../requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Aarush005coder/Machine-Learning-Notebooks.git
```

### 2. Navigate to the KNN Folder

```bash
cd Machine-Learning-Notebooks/05_K_Nearest_Neighbors
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the Notebook

Start with:

```text
K_Nearest_Neighbors_01_Basics.ipynb
```

### 5. Run the Cells

Run the notebook cells sequentially and explore the generated:

* **Outputs**
* **Graphs**
* **Predictions**
* **Confusion Matrix**
* **K vs Accuracy Plot**
* **Decision Boundaries**
* **Classification Metrics**

---

## 📁 Folder Structure

```text
05_K_Nearest_Neighbors/
│
├── K_Nearest_Neighbors_01_Basics.ipynb
└── README.md
```

---

## 🔑 Key Concepts

| **Concept**            | **Description**                                              |
| ---------------------- | ------------------------------------------------------------ |
| **KNN**                | Supervised learning algorithm based on nearest observations  |
| **K**                  | Number of neighbors considered for prediction                |
| **Neighbor**           | Training observation closest to a new observation            |
| **Distance**           | Measure used to determine similarity between observations    |
| **Euclidean Distance** | Straight-line distance between points                        |
| **Manhattan Distance** | Sum of absolute coordinate differences                       |
| **Minkowski Distance** | Generalized distance metric                                  |
| **Feature Scaling**    | Transforming features to comparable scales                   |
| **Majority Voting**    | Selecting the most common class among neighbors              |
| **Classification**     | Predicting a categorical target                              |
| **Regression**         | Predicting a continuous numerical target                     |
| **Decision Boundary**  | Region separating different predicted classes                |
| **Accuracy**           | Proportion of correctly classified observations              |
| **Precision**          | Proportion of predicted positives that are actually positive |
| **Recall**             | Proportion of actual positives correctly identified          |
| **F1-Score**           | Harmonic mean of precision and recall                        |
| **Overfitting**        | Model becomes overly sensitive to training observations      |
| **Underfitting**       | Model is too simple to capture useful patterns               |

---

## 📌 Learning Path

```text
K-Nearest Neighbors
        ↓
Supervised Learning
        ↓
Classification & Regression
        ↓
Distance Metrics
        ↓
Feature Scaling
        ↓
Train-Test Split
        ↓
Choose K
        ↓
Model Training
        ↓
Nearest Neighbors
        ↓
Prediction
        ↓
Accuracy / Precision / Recall / F1
        ↓
Decision Boundary
        ↓
K Selection
        ↓
Model Interpretation
```

---

## 📂 Repository Structure

The main repository contains multiple machine learning algorithms and concepts.

```text
Machine-Learning-Notebooks/
│
├── 01_Linear_Regression/
│
├── 02_Multiple_Linear_Regression/
│
├── 03_Polynomial_Regression/
│
├── 04_Logistic_Regression/
│
├── 05_K_Nearest_Neighbors/
│   ├── K_Nearest_Neighbors_01_Basics.ipynb
│   └── README.md
│
├── 06_Decision_Tree/
│
├── 07_Random_Forest/
│
├── 08_Support_Vector_Machine/
│
├── 09_Naive_Bayes/
│
├── 10_K_Means_Clustering/
│
├── 11_Hierarchical_Clustering/
│
├── 12_PCA/
│
├── 13_Gradient_Boosting/
│
├── 14_XGBoost/
│
├── datasets/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 💡 Key Takeaways

* **KNN is a supervised machine learning algorithm based on neighboring observations.**
* **KNN can be used for both classification and regression.**
* **The value of K determines how many neighbors influence a prediction.**
* **Distance metrics determine how similarity between observations is measured.**
* **Feature scaling is important because KNN relies on distance calculations.**
* **Small K values can make the model more sensitive to local observations.**
* **Large K values can produce smoother predictions.**
* **KNN classification commonly uses majority voting.**
* **KNN regression commonly uses neighboring numerical values to estimate the prediction.**
* **Accuracy, precision, recall, and F1-score can be used to evaluate classification performance.**
* **KNN can produce flexible decision boundaries.**
* **High dimensionality and large datasets can make KNN computationally expensive.**

---

## 🎓 Purpose of This Section

The goal of this section is to build a strong foundation in **distance-based machine learning** through practical implementation.

The focus is not only on using:

```python
KNeighborsClassifier()
```

but on understanding the complete KNN workflow:

```text
Understand the Data
       ↓
Scale the Features
       ↓
Choose K
       ↓
Calculate Distances
       ↓
Find Neighbors
       ↓
Make Predictions
       ↓
Evaluate Performance
       ↓
Tune K
       ↓
Interpret Results
```

---

## 🔗 Part of the Machine Learning Journey

This KNN section is part of the **Machine-Learning-Notebooks** repository, which contains practical implementations of machine learning algorithms using Python and Jupyter Notebooks.

> **Learn the concept → Understand the distance → Implement it → Visualize it → Evaluate it → Tune it → Understand it**
