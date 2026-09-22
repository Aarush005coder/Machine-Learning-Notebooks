# Linear Regression

A practical and beginner-friendly collection of **Jupyter Notebooks** covering **Linear Regression**, one of the most fundamental supervised machine learning algorithms used for predicting continuous numerical values.

This section focuses on understanding the core concepts, implementing Linear Regression using **Python and Scikit-learn**, training models, making predictions, visualizing regression lines, and evaluating regression performance using standard metrics.

---

## 📚 Contents

|  **#** | **Notebook**                                                               | **Description**                                                                                                                                                                        |
| -----: | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **01** | [`Linear_Regression_01_Basics.ipynb`](./Linear_Regression_01_Basics.ipynb) | Fundamentals of Linear Regression, feature and target selection, train-test split, model training, predictions, regression line, residual analysis, and regression evaluation metrics. |

---

## 🧠 What is Linear Regression?

**Linear Regression** is a **supervised machine learning algorithm** used primarily for **regression problems**, where the target variable is continuous.

It attempts to model the relationship between input features and a continuous target variable using a linear equation.

For **Simple Linear Regression**:

```text
y = β₀ + β₁x
```

Where:

* **y** → Predicted output
* **x** → Input feature
* **β₀** → Intercept
* **β₁** → Coefficient / Slope

The objective is to find the line that best represents the relationship between the input and target variables.

---

## 📈 Simple Linear Regression

Simple Linear Regression uses **one independent variable** to predict a dependent variable.

### Example

In the notebook:

```text
Input Feature → Years of Experience
Target        → Salary
```

The model learns a relationship between experience and salary and can then predict salary for previously unseen values.

### Example Relationship

```text
Salary
  ↑
  │                         ●
  │                    ●
  │                 ●
  │             ●
  │         ●
  │      ●
  │   ●
  │ ●
  └────────────────────────────→ Years of Experience
```

The straight line represents the **regression line** learned by the model.

---

## 🔢 How Linear Regression Works

Linear Regression tries to find the values of the coefficients that minimize the difference between the actual and predicted values.

The prediction equation is:

```text
ŷ = β₀ + β₁x
```

For multiple features, the equation becomes:

```text
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

The model learns the coefficients during training.

---

## 📉 Best-Fit Line

The **best-fit line** is the line that minimizes the overall prediction error between the observed data points and the values predicted by the line.

The vertical difference between an actual observation and its predicted value is called a **residual**.

```text
Actual Point
     ●
     │
     │  Residual
     │
─────┼────────────────── Regression Line
     │
     └────────────────────────→ X
```

A good regression model attempts to keep these errors relatively small while capturing the underlying relationship in the data.

---

## 🎯 Regression Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Visualization
   ↓
Feature & Target Selection
   ↓
Train-Test Split
   ↓
Create Linear Regression Model
   ↓
Model Training
   ↓
Predictions
   ↓
Model Evaluation
   ↓
Residual Analysis
   ↓
Interpretation
```

---

## 🔀 Train-Test Split

The dataset is divided into two parts:

```text
Dataset
   │
   ├───────────────┐
   ↓               ↓
Training Data   Testing Data
   │               │
   ↓               ↓
Train Model     Evaluate Model
```

The **training set** is used to learn the model parameters.

The **testing set** is used to evaluate how well the trained model performs on unseen data.

In the notebook, `train_test_split()` from Scikit-learn is used for this purpose.

---

## 📊 Model Evaluation

The notebook uses several important regression metrics to evaluate model performance.

### 1. Mean Absolute Error — MAE

MAE measures the average absolute difference between actual and predicted values.

```text
MAE = Average(|Actual - Predicted|)
```

A lower MAE indicates smaller average prediction errors.

---

### 2. Mean Squared Error — MSE

MSE calculates the average squared difference between actual and predicted values.

```text
MSE = Average((Actual - Predicted)²)
```

Squaring the errors gives larger errors greater influence on the metric.

---

### 3. Root Mean Squared Error — RMSE

RMSE is the square root of MSE.

```text
RMSE = √MSE
```

RMSE is expressed in the same units as the target variable, making it easier to interpret than MSE.

---

### 4. R² Score

The **R² score**, also known as the coefficient of determination, measures how much of the variation in the target variable is explained by the regression model.

The score can be interpreted in relation to the baseline used for the metric, and values closer to 1 indicate that the model explains a larger proportion of the observed variation in the evaluated data.

---

## 📐 Residual Analysis

A **residual** is the difference between the actual value and the predicted value.

```text
Residual = Actual - Predicted
```

For example:

```text
Actual Salary     = 60000
Predicted Salary  = 57000

Residual = 60000 - 57000
         = 3000
```

Residual analysis helps us understand the errors made by the regression model.

### Residual Plot

```text
Residual
   ↑
 + │     ●       ●
   │  ●      ●
 0 ├──────────────────────
   │     ●    ●      ●
 - │ ●       ●
   └──────────────────────→ Predicted Value
```

A residual plot can help identify patterns that may indicate that a simple linear model is not adequately capturing the relationship in the data.

---

## 📈 Visualizations

The notebook includes visualizations such as:

* **Scatter Plot**
* **Regression Line**
* **Actual vs Predicted Plot**
* **Residual Plot**
* **Feature vs Target Relationship**

These visualizations help understand the relationship between variables and the behavior of the trained model.

---

## 🧪 Dataset Used

The basics notebook uses a practical **Salary vs Years of Experience** dataset.

### Features

```text
YearsExperience
```

### Target

```text
Salary
```

### Objective

```text
Years of Experience
          ↓
   Linear Regression
          ↓
    Salary Prediction
```

The model learns the relationship from the training data and predicts salary values for unseen observations.

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
git clone <repository-url>
```

### 2. Navigate to the Linear Regression Folder

```bash
cd Machine-Learning-Notebooks/01_Linear_Regression
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the Notebook

Start with:

```text
Linear_Regression_01_Basics.ipynb
```

### 5. Run the Cells

Run the notebook cells sequentially and explore the generated:

* **Outputs**
* **Graphs**
* **Predictions**
* **Regression Line**
* **Evaluation Metrics**
* **Residual Analysis**

---

## 📁 Folder Structure

```text
01_Linear_Regression/
│
├── Linear_Regression_01_Basics.ipynb
└── README.md

```

## 🎯 Learning Objectives

After completing this notebook, you should be able to:

* **Explain Linear Regression**
* **Understand regression problems**
* **Differentiate features and targets**
* **Understand slope and intercept**
* **Understand the regression equation**
* **Visualize relationships between numerical variables**
* **Split data into training and testing sets**
* **Train Linear Regression using Scikit-learn**
* **Make predictions on unseen data**
* **Interpret model coefficients**
* **Evaluate regression performance**
* **Understand MAE, MSE, RMSE, and R²**
* **Understand residuals**
* **Analyze model predictions using visualizations**

---

## 🌍 Real-World Applications

Linear Regression can be used for various regression problems, including:

* **Salary Prediction**
* **House Price Prediction**
* **Sales Forecasting**
* **Revenue Estimation**
* **Demand Prediction**
* **Cost Estimation**
* **Temperature Prediction**
* **Business Trend Analysis**
* **Performance Prediction**

The suitability of Linear Regression depends on the data, the relationship between variables, and the assumptions of the model.

---

## ⚠️ Important Assumptions

Linear Regression generally works best when important assumptions are reasonably satisfied.

### 1. Linearity

The relationship between the predictors and target should be approximately linear.

### 2. Independence

Observations should be reasonably independent of each other.

### 3. Homoscedasticity

The variance of residuals should be reasonably consistent across the range of predictions.

### 4. Low Multicollinearity

For Multiple Linear Regression, predictors should not exhibit problematic levels of correlation with one another.

### 5. Residual Distribution

The distribution of residuals can be important, particularly when performing statistical inference.

---

## 🔑 Key Concepts

| **Concept**           | **Description**                                                         |
| --------------------- | ----------------------------------------------------------------------- |
| **Regression**        | Predicting a continuous numerical value                                 |
| **Linear Regression** | Models a linear relationship between predictors and a continuous target |
| **Feature**           | Input variable used by the model                                        |
| **Target**            | Variable the model attempts to predict                                  |
| **Coefficient**       | Represents the effect of a feature in the fitted linear model           |
| **Intercept**         | Model output when the input features are zero                           |
| **Prediction**        | Output generated by the trained model                                   |
| **Residual**          | Difference between actual and predicted values                          |
| **MAE**               | Average absolute prediction error                                       |
| **MSE**               | Average squared prediction error                                        |
| **RMSE**              | Square root of MSE                                                      |
| **R² Score**          | Measures explained variation relative to a baseline                     |

---

## 📌 Learning Path

```text
Linear Regression
        ↓
Regression Problems
        ↓
Features & Target
        ↓
Regression Equation
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Prediction
        ↓
MAE / MSE / RMSE
        ↓
R² Score
        ↓
Residual Analysis
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
│   ├── Linear_Regression_01_Basics.ipynb
│   └── README.md
│
├── 02_Multiple_Linear_Regression/
│
├── 03_Polynomial_Regression/
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
├── datasets/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 💡 Key Takeaways

* **Linear Regression is a supervised learning algorithm used for regression problems.**
* **It predicts continuous numerical values.**
* **Simple Linear Regression uses one input feature.**
* **The model learns a linear relationship between the input and target.**
* **The slope represents the learned change associated with a feature.**
* **The intercept represents the model's baseline output when the input features are zero.**
* **Train-test splitting helps evaluate the model on unseen data.**
* **MAE, MSE, RMSE, and R² are important regression evaluation metrics.**
* **Residual analysis helps understand model errors and potential patterns.**
* **Visualization makes it easier to understand the fitted model and its predictions.**

---

## 🎓 Purpose of This Section

The goal of this section is to build a strong foundation in **regression-based machine learning** through practical implementation.

The focus is not only on using:

```python
LinearRegression()
```

but on understanding the complete machine learning workflow:

```text
Understand the Data
       ↓
Build the Model
       ↓
Train the Model
       ↓
Make Predictions
       ↓
Evaluate Performance
       ↓
Analyze Errors
       ↓
Interpret Results
```

---

## 🔗 Part of the Machine Learning Journey

This Linear Regression section is part of the **Machine-Learning-Notebooks** repository, which contains practical implementations of machine learning algorithms using Python and Jupyter Notebooks.

> **Learn the concept → Implement it → Visualize it → Evaluate it → Understand it**
