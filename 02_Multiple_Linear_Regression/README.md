# 📈 Multiple Linear Regression

Multiple Linear Regression is a supervised machine learning algorithm used to predict a **continuous target variable using two or more independent variables**.

It extends Simple Linear Regression by allowing the model to learn the relationship between multiple input features and a single output.

For example, predicting **house price** using:

* Area
* Number of bedrooms
* Number of bathrooms
* Age of the house

Multiple Linear Regression learns how each feature contributes to the final prediction.

---

## 📚 Contents

| **#NotebookDescription** |                                                                                                                                                                                        |                                                                                                                                                                                                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **01**                   | [`Multiple_Linear_Regression_01_Basics.ipynb`](https://github.com/Aarush005coder/Machine-Learning-Notebooks/blob/main/02_Multiple_Linear_Regression/Multiple_Linear_Regression_01_Basics.ipynb) | Fundamentals of Multiple Linear Regression, multiple feature selection, target selection, train-test split, model training, coefficients, intercept, predictions, residual analysis, and regression evaluation metrics.<br><br> |

---

## 🔹 What is Multiple Linear Regression?

Multiple Linear Regression is a supervised learning algorithm used for **regression problems** where multiple independent variables are used to predict one continuous dependent variable.

Unlike Simple Linear Regression, which uses only one feature:

```text
X → Y
```

Multiple Linear Regression uses multiple features:

```text
X₁
X₂
X₃  ───→ Y
X₄
```

For example:

```text
Area ──────────────┐
Bedrooms ──────────┤
Bathrooms ─────────┼──→ House Price
Location Score ────┤
Age ───────────────┘
```

The model learns the contribution of each feature and combines them to produce the final prediction.

---

## 🔹 Multiple Linear Regression Equation

The general equation is:

```text
ŷ = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + ... + βₙXₙ
```

Where:

* `ŷ` = Predicted output
* `β₀` = Intercept
* `β₁, β₂, ..., βₙ` = Model coefficients
* `X₁, X₂, ..., Xₙ` = Input features
* `n` = Number of features

For three features:

```text
ŷ = β₀ + β₁X₁ + β₂X₂ + β₃X₃
```

The model finds the values of the coefficients that minimize the prediction error.

---

## 🔹 How Multiple Linear Regression Works

The basic process is:

```text
Dataset
   ↓
Select Features and Target
   ↓
Train-Test Split
   ↓
Train Multiple Linear Regression Model
   ↓
Learn Coefficients
   ↓
Make Predictions
   ↓
Evaluate Model
```

The model tries to find the best combination of coefficients that minimizes the difference between actual and predicted values.

The most common approach is **Ordinary Least Squares (OLS)**.

---

## 🔹 Model Coefficients

Each feature receives a coefficient.

For example:

```text
Price = 50000
       + 200 × Area
       + 10000 × Bedrooms
       - 1500 × Age
```

Here:

```text
Area      → +200
Bedrooms  → +10000
Age       → -1500
```

A coefficient represents the expected change in the target when that feature increases by one unit, **while keeping the other features constant**.

### Positive Coefficient

```text
β > 0
```

Generally indicates a positive relationship with the target.

### Negative Coefficient

```text
β < 0
```

Generally indicates a negative relationship with the target.

---

## 🔹 Intercept

The intercept is represented by:

```text
β₀
```

It is the predicted value of the target when all input features are zero.

For example:

```text
ŷ = 100 + 5X₁ + 10X₂
```

Here:

```text
Intercept = 100
```

In real-world datasets, the intercept may not always have a meaningful practical interpretation, especially when zero values for the features are unrealistic.

---

## 🔹 Train-Test Split

The dataset is divided into:

### Training Data

Used to learn:

```text
Coefficients
Intercept
```

### Testing Data

Used to evaluate how well the trained model performs on unseen data.

Example:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Here:

```text
80% → Training
20% → Testing
```

---

## 🔹 Model Training

Scikit-learn provides:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

The model learns:

```text
β₀
β₁
β₂
...
βₙ
```

These parameters define the regression equation.

---

## 🔹 Prediction

After training:

```python
y_pred = model.predict(X_test)
```

For a new observation:

```python
new_prediction = model.predict([[X1, X2, X3]])
```

The model substitutes the feature values into the learned regression equation.

---

## 🔹 Model Evaluation

Common regression metrics include:

### Mean Absolute Error

```text
MAE = (1/n) Σ |yᵢ - ŷᵢ|
```

Measures the average absolute prediction error.

---

### Mean Squared Error

```text
MSE = (1/n) Σ (yᵢ - ŷᵢ)²
```

Penalizes larger errors more heavily.

---

### Root Mean Squared Error

```text
RMSE = √MSE
```

RMSE is expressed in the same units as the target variable.

---

### R² Score

```text
R² = 1 - SSres / SStot
```

R² measures how much of the variation in the target is explained by the model.

Example:

```text
R² = 0.85
```

means that approximately 85% of the variation in the target is explained by the model on that evaluated dataset.

---

## 🔹 Residual Analysis

A residual is the difference between the actual and predicted value:

```text
Residual = Actual - Predicted
```

or:

```text
eᵢ = yᵢ - ŷᵢ
```

Residual analysis helps identify problems such as:

* Non-linearity
* Outliers
* Heteroscedasticity
* Model misspecification

Ideally, residuals should be randomly distributed around zero.

---

## 🔹 Feature Importance

Multiple Linear Regression does not provide feature importance in exactly the same way as tree-based models.

However, coefficients can help understand feature relationships.

For example:

```python
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
```

A coefficient should be interpreted carefully because its magnitude depends on the feature's units and scale.

---

## 🔹 Visualizations

Multiple Linear Regression can be explored using:

* Actual vs Predicted Plot
* Residual Plot
* Residual Distribution
* Feature vs Target Plots
* Correlation Heatmap
* Coefficient Plot

An actual-vs-predicted plot helps compare:

```text
Actual Values
      vs
Predicted Values
```

A good model generally has predictions close to the ideal:

```text
Actual = Predicted
```

---

## 🔹 Dataset Used

The notebook uses a dataset containing multiple independent variables and one continuous target variable.

Example structure:

| Feature     | Description                   |
| ----------- | ----------------------------- |
| `Feature 1` | First independent variable    |
| `Feature 2` | Second independent variable   |
| `Feature 3` | Third independent variable    |
| `Target`    | Continuous dependent variable |

The dataset is used to demonstrate how multiple features can jointly predict a target.

---

## 🛠️ Technologies & Libraries

The notebook uses:

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Aarush005coder/Machine-Learning-Notebooks.git
```

Navigate to the repository:

```bash
cd Machine-Learning-Notebooks
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Start Jupyter Notebook:

```bash
jupyter notebook
```

Navigate to:

```text
02_Multiple_Linear_Regression/
```

Open:

```text
Multiple_Linear_Regression_01_Basics.ipynb
```

Run the notebook cells sequentially.

---

## 📁 Folder Structure

```text
02_Multiple_Linear_Regression/
│
├── Multiple_Linear_Regression_01_Basics.ipynb
│
└── README.md
```

---

## 🧪 Notebook 01 — Multiple Linear Regression Basics

### `Multiple_Linear_Regression_01_Basics.ipynb`

This notebook introduces the fundamental concepts of Multiple Linear Regression and demonstrates how multiple independent variables can be used to predict a continuous target variable.

### Topics Covered

* **Introduction to Multiple Linear Regression**
* **Supervised Learning**
* **Regression**
* **Dependent and Independent Variables**
* **Multiple Features**
* **Dataset Exploration**
* **Feature and Target Selection**
* **Train-Test Split**
* **Ordinary Least Squares**
* **Model Training**
* **Intercept**
* **Regression Coefficients**
* **Predictions**
* **Mean Absolute Error**
* **Mean Squared Error**
* **Root Mean Squared Error**
* **R² Score**
* **Residuals**
* **Residual Analysis**
* **Actual vs Predicted Values**
* **Correlation Analysis**
* **Feature Relationships**
* **Coefficient Interpretation**
* **Model Performance**
* **Outlier Analysis**
* **Multiple Linear Regression Assumptions**

---

## 🎯 Learning Objectives

After completing this section, you should understand:

* What Multiple Linear Regression is
* Difference between Simple and Multiple Linear Regression
* How multiple features are used for prediction
* Regression equations
* Intercept and coefficients
* Ordinary Least Squares
* Train-test splitting
* Model training
* Making predictions
* MAE, MSE and RMSE
* R² score
* Residual analysis
* Coefficient interpretation
* Regression assumptions
* Model evaluation

---

## 🌍 Real-World Applications

Multiple Linear Regression is commonly used in:

* 🏠 House Price Prediction
* 💰 Sales Prediction
* 📊 Revenue Forecasting
* 📈 Business Analytics
* 🏭 Production Estimation
* 🚗 Vehicle Price Prediction
* 🏦 Financial Analysis
* 📦 Demand Forecasting
* 🏥 Cost Estimation
* 📢 Marketing Analytics

For example:

```text
Advertising Budget
        +
Product Price
        +
Season
        +
Previous Sales
        ↓
    Sales Prediction
```

---

## ⚠️ Important Assumptions

Multiple Linear Regression commonly relies on several assumptions:

### 1. Linearity

The relationship between predictors and the target should be approximately linear.

### 2. Independence

Observations should generally be independent.

### 3. Homoscedasticity

The variance of residuals should remain reasonably constant.

### 4. Normality of Residuals

For statistical inference, residuals are often assumed to be approximately normally distributed.

### 5. Low Multicollinearity

Independent variables should not have extremely strong linear relationships with each other.

### 6. No Severe Outliers

Extreme observations can strongly influence the fitted model.

---

## 🔑 Key Concepts

Important concepts covered in this section:

```text
Multiple Linear Regression
        ↓
Independent Variables
        ↓
Dependent Variable
        ↓
Regression Equation
        ↓
Coefficients
        ↓
Intercept
        ↓
Ordinary Least Squares
        ↓
Predictions
        ↓
Residuals
        ↓
Model Evaluation
```

---

## 🧭 Learning Path

Recommended progression:

```text
Simple Linear Regression
        ↓
Multiple Linear Regression
        ↓
Polynomial Regression
        ↓
Regularization
        ↓
Ridge Regression
        ↓
Lasso Regression
        ↓
Elastic Net
```

Understanding Multiple Linear Regression provides an important foundation for more advanced regression techniques.

---

## 📂 Repository Structure

```text
Machine-Learning-Notebooks/
│
├── 01_Linear_Regression/
│   ├── Linear_Regression_01_Basics.ipynb
│   └── README.md
│
├── 02_Multiple_Linear_Regression/
│   ├── Multiple_Linear_Regression_01_Basics.ipynb
│   └── README.md
│
├── 03_Polynomial_Regression/
│   ├── Polynomial_Regression_01_Basics.ipynb
│   └── README.md
│
├── ...
│
├── datasets/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 💡 Key Takeaways

* Multiple Linear Regression uses **multiple independent variables** to predict one continuous target.
* The model learns an intercept and multiple coefficients.
* The general equation is:

```text
ŷ = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
```

* Coefficients describe the relationship between individual features and the target while holding other features constant.
* MAE, MSE, RMSE and R² can be used to evaluate regression performance.
* Residual analysis helps identify potential model problems.
* Multicollinearity can make coefficient interpretation difficult.
* Multiple Linear Regression is an important foundation for advanced regression algorithms.

---

## 🎯 Purpose of This Section

The purpose of this section is to build a strong understanding of **Multiple Linear Regression**, from mathematical foundations to practical implementation.

The notebook focuses on:

```text
Theory
   ↓
Dataset
   ↓
Feature Selection
   ↓
Model Training
   ↓
Prediction
   ↓
Evaluation
   ↓
Interpretation
```

This creates a foundation for more advanced regression algorithms and machine learning models.

---

## 🚀 Part of the Machine Learning Journey

This section is part of the **Machine Learning Notebooks** repository and represents the next step after Simple Linear Regression.

```text
Machine Learning
       │
       └── Supervised Learning
              │
              └── Regression
                     │
                     ├── Simple Linear Regression
                     │
                     ├── Multiple Linear Regression
                     │
                     ├── Polynomial Regression
                     │
                     ├── Ridge Regression
                     │
                     ├── Lasso Regression
                     │
                     └── Elastic Net
```

The goal is to develop a strong understanding of regression algorithms through **theory, implementation, visualization, evaluation, and practical examples**.
