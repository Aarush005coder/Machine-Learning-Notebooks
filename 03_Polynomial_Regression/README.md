# Polynomial Regression

Polynomial Regression is a supervised machine learning regression technique used to model **non-linear relationships** between independent and dependent variables.

Unlike Simple Linear Regression, which fits a straight line, Polynomial Regression transforms the input features into polynomial features such as \(X^2\), \(X^3\), etc., allowing the model to capture curved relationships in the data.

---

## 📚 Contents

| **#** | Notebook                                | Description                                                                                |
|-------| --------------------------------------- | ------------------------------------------------------------------------------------------ |
|   01  | `Polynomial_Regression_01_Basics.ipynb` | Complete implementation of Polynomial Regression from data preparation to model evaluation |

---

## 🧠 What is Polynomial Regression?

Polynomial Regression extends Linear Regression by adding polynomial terms of the input features.

### Linear Regression

$$
y = \beta_0 + \beta_1X
$$

This produces a straight line.

### Polynomial Regression

$$
y = \beta_0 + \beta_1X + \beta_2X^2 + \beta_3X^3 + \cdots + \beta_nX^n
$$

This allows the model to represent non-linear and curved relationships.

For example, a degree-2 polynomial contains:

```text
X
X²
```

A degree-3 polynomial contains:

```text
X
X²
X³
```

---

## 🔄 How Polynomial Regression Works

The basic workflow is:

```text
Original Features
       ↓
Polynomial Feature Transformation
       ↓
X, X², X³, ...
       ↓
Linear Regression Model
       ↓
Predictions
       ↓
Model Evaluation
```

Polynomial Regression is still a **linear regression model with respect to its parameters**. The non-linearity comes from transforming the input features into polynomial terms.

---

## ⚙️ Important Parameters

### 1. Degree

The `degree` determines the highest polynomial power used by the model.

```python
PolynomialFeatures(degree=2)
```

creates:

```text
1, X, X²
```

While:

```python
PolynomialFeatures(degree=3)
```

creates:

```text
1, X, X², X³
```

A higher degree can make the model more flexible but can also increase the risk of overfitting.

---

### 2. PolynomialFeatures

Scikit-learn provides `PolynomialFeatures` for creating polynomial features.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)
```

---

### 3. LinearRegression

After transforming the features, a standard Linear Regression model can be trained.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_poly, y)
```

---

## 📊 Example Dataset

The notebook uses a **Position Level vs Salary** dataset.

Example:

| Position Level | Salary  |
| -------------- | ------- |
| 1              | 45,000  |
| 2              | 50,000  |
| 3              | 60,000  |
| 4              | 80,000  |
| 5              | 110,000 |
| 6              | 150,000 |
| 7              | 200,000 |
| 8              | 270,000 |
| 9              | 350,000 |
| 10             | 450,000 |

The relationship between position level and salary is non-linear, making it suitable for demonstrating Polynomial Regression.

---

## 💻 Implementation

### Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
```

### Create Polynomial Features

```python
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)
```

### Train Model

```python
model = LinearRegression()

model.fit(X_poly, y)
```

### Make Predictions

```python
y_pred = model.predict(X_poly)
```

---

## 📈 Visualization

The notebook includes visualization of:

* Original data points
* Linear Regression
* Polynomial Regression curve
* Different polynomial degrees
* Actual vs Predicted values
* Residuals

Polynomial Regression can represent curved patterns that a straight-line model cannot capture.

---

## 📐 Polynomial Regression Equation

For degree 2:

$$
y = \beta_0 + \beta_1X + \beta_2X^2
$$

For degree 3:

$$
y = \beta_0 + \beta_1X + \beta_2X^2 + \beta_3X^3
$$

General form:

$$
y = \beta_0 + \beta_1X + \beta_2X^2 + \cdots + \beta_nX^n
$$

Where:

* \(y\) = predicted output
* \(X\) = input feature
* \(\beta_0\) = intercept
* \(\beta_1, \beta_2, \ldots, \beta_n\) = model coefficients
* \(n\) = polynomial degree

---

## 📏 Model Evaluation

The notebook evaluates the model using:

### Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n}\sum |y_i-\hat{y}_i|
$$

Measures the average absolute prediction error.

### Mean Squared Error (MSE)

$$
MSE = \frac{1}{n}\sum(y_i-\hat{y}_i)^2
$$

Penalizes larger errors more heavily.

### Root Mean Squared Error (RMSE)

$$
RMSE = \sqrt{MSE}
$$

Expresses the error in the same unit as the target variable.

### R² Score

$$
R^2 = 1-\frac{SS_{res}}{SS_{tot}}
$$

Measures how much of the variation in the target variable is explained by the model.

---

## ⚖️ Polynomial Degree and Overfitting

The polynomial degree is an important hyperparameter.

### Low Degree

```text
Degree = 1
```

The model behaves like Linear Regression and may underfit non-linear data.

### Appropriate Degree

```text
Degree = 2
Degree = 3
```

Can capture the underlying curved relationship.

### Very High Degree

```text
Degree = 10
Degree = 20
...
```

Can create an overly complex curve and lead to **overfitting**.

```text
Low Degree
    ↓
Underfitting

Appropriate Degree
    ↓
Good Fit

Very High Degree
    ↓
Overfitting
```

---

## 🔬 Linear vs Polynomial Regression

| Feature                | Linear Regression    | Polynomial Regression    |
| ---------------------- | -------------------- | ------------------------ |
| Relationship           | Linear               | Non-linear               |
| Output shape           | Straight line        | Curve                    |
| Features               | X                    | X, X², X³, ...           |
| Complexity             | Low                  | Higher                   |
| Overfitting Risk       | Lower                | Higher with large degree |
| Feature Transformation | Not required         | Required                 |
| Use Case               | Linear relationships | Curved relationships     |

---

## 📊 Visualizations Included

The notebook contains:

1. Original Data Distribution
2. Linear Regression Fit
3. Polynomial Regression Fit
4. Polynomial Curves for Different Degrees
5. Actual vs Predicted Values
6. Residual Plot
7. Feature Coefficient Visualization

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## 📦 Installation

Install the required dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Or install all repository dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/Aarush005coder/Machine-Learning-Notebooks.git
```

Navigate to the project:

```bash
cd Machine-Learning-Notebooks
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
03_Polynomial_Regression/
└── Polynomial_Regression_01_Basics.ipynb
```

Run the notebook cells sequentially.

---

## 📁 Folder Structure

```text
03_Polynomial_Regression/
│
├── Polynomial_Regression_01_Basics.ipynb
└── README.md
```

---

## 🎯 Learning Objectives

After completing this notebook, you should understand:

* What Polynomial Regression is
* Why Linear Regression may not fit non-linear data
* How polynomial features are generated
* How `PolynomialFeatures` works
* How to select polynomial degree
* How Polynomial Regression uses Linear Regression internally
* How to train a Polynomial Regression model
* How to make predictions
* How to evaluate regression models
* How polynomial degree affects underfitting and overfitting
* How to visualize polynomial regression curves
* How to analyze residuals

---

## 🌍 Real-World Applications

Polynomial Regression can be useful when the relationship between variables is curved or non-linear.

Examples include:

* Salary and experience modeling
* Temperature and energy consumption
* Population growth modeling
* Business revenue analysis
* Manufacturing processes
* Economic trend analysis
* Performance vs experience
* Demand forecasting
* Sensor calibration
* Scientific data modeling

---

## ⚠️ Important Considerations

### 1. Choosing the Degree

The polynomial degree should be selected carefully.

A very low degree may underfit, while a very high degree may overfit.

### 2. Feature Scaling

For high-degree polynomial features, feature values can become very large.

Feature scaling may therefore become important.

### 3. Multicollinearity

Polynomial features such as:

```text
X
X²
X³
X⁴
```

can be highly correlated with each other.

### 4. Computational Complexity

The number of generated features can increase rapidly when using multiple input features and high polynomial degrees.

---

## 🧩 Key Concepts

```text
Polynomial Regression
│
├── Polynomial Features
│   ├── X
│   ├── X²
│   ├── X³
│   └── Xⁿ
│
├── Polynomial Degree
│
├── Linear Regression
│
├── Model Coefficients
│
├── Predictions
│
├── Residuals
│
├── MAE
├── MSE
├── RMSE
└── R² Score
```

---

## 📚 Learning Path

Recommended progression:

```text
01. Simple Linear Regression
        ↓
02. Multiple Linear Regression
        ↓
03. Polynomial Regression
        ↓
04. Regularization
        ↓
05. Ridge Regression
        ↓
06. Lasso Regression
        ↓
07. Elastic Net
```

---

## 📂 Repository Structure

```text
Machine-Learning-Notebooks/
│
├── 01_Linear_Regression/
│
├── 02_Multiple_Linear_Regression/
│
├── 03_Polynomial_Regression/
│   ├── Polynomial_Regression_01_Basics.ipynb
│   └── README.md
│
├── 04_Logistic_Regression/
│
├── 05_K_Nearest_Neighbors/
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
├── 15_DBSCAN/
│
├── datasets/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🔑 Key Takeaways

* Polynomial Regression is used for non-linear relationships.
* It creates polynomial features such as \(X^2\), \(X^3\), etc.
* `PolynomialFeatures` performs the feature transformation.
* A Linear Regression model is then trained on the transformed features.
* The polynomial degree controls model complexity.
* A low degree can cause underfitting.
* A very high degree can cause overfitting.
* MAE, MSE, RMSE, and R² can be used to evaluate the model.
* Visualization helps understand how well the polynomial curve fits the data.

---

## 🎯 Purpose

This notebook is part of my **Machine Learning journey** and focuses on understanding Polynomial Regression from fundamentals to practical implementation.

The goal is to build a strong understanding of regression algorithms through:

* Mathematical concepts
* Python implementation
* Data visualization
* Model evaluation
* Practical examples
* Understanding model behavior

---

## 🚀 Machine Learning Journey

This notebook is part of the **Machine-Learning-Notebooks** repository, where different machine learning algorithms are implemented and studied progressively using Python and Scikit-learn.
