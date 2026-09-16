# Logistic Regression

A practical and beginner-friendly collection of **Jupyter Notebooks** covering **Logistic Regression**, one of the most widely used supervised machine learning algorithms for classification.

This section focuses on understanding the core concepts, implementing Logistic Regression using **Python and Scikit-learn**, evaluating classification performance, visualizing decision boundaries, and handling **imbalanced datasets**.

---

## 📚 Contents

| #      | Notebook                                                                                         | Description                                                                                                                                                                 |
| ------ | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **01** | [`Logistic_Regression_01_Basics.ipynb`](./Logistic_Regression_01_Basics.ipynb)                   | Fundamentals of Logistic Regression, binary classification, sigmoid function, model training, predictions, evaluation metrics, ROC curve, ROC-AUC, and decision boundaries. |
| **02** | [`Logistic_Regression_02_Imbalanced_Data.ipynb`](./Logistic_Regression_02_Imbalanced_Data.ipynb) | Logistic Regression with imbalanced datasets using Random UnderSampling and Random OverSampling, along with ROC-AUC and decision boundary analysis.                         |

---

## 🧠 What is Logistic Regression?

**Logistic Regression** is a **supervised machine learning algorithm** primarily used for **classification problems**.

Instead of directly predicting a continuous numerical value, Logistic Regression estimates the **probability** that an observation belongs to a particular class.

For binary classification, the two classes are commonly represented as:

```text
Class 0
Class 1
```

### Examples

* **Spam vs Not Spam**
* **Fraud vs Not Fraud**
* **Pass vs Fail**
* **Customer Churn vs No Churn**
* **Disease vs No Disease**

Despite its name, **Logistic Regression is primarily a classification algorithm**.

---

## 🔢 How Logistic Regression Works

Logistic Regression first calculates a linear combination of the input features:

```text
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

This value is passed through the **Sigmoid Function** to convert it into a probability between **0 and 1**.

### Sigmoid Function

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

The sigmoid function produces an output in the range:

```text
0 ≤ σ(z) ≤ 1
```

A typical binary classification rule is:

```text
Probability < 0.5  →  Class 0
Probability ≥ 0.5  →  Class 1
```

The classification threshold can be adjusted depending on the requirements of the problem.

---

## 🎯 Binary Classification

Logistic Regression is commonly used for **binary classification**, where the model predicts one of two classes.

### Classification Workflow

```text
Input Features
      ↓
Linear Combination
      ↓
Sigmoid Function
      ↓
Probability
      ↓
Decision Threshold
      ↓
Class Prediction
```

### Example

If the model produces:

```text
P(Class 1) = 0.82
```

Using a threshold of `0.50`:

```text
0.82 ≥ 0.50
      ↓
Prediction = Class 1
```

---

## ⚖️ Linear Regression vs Logistic Regression

| Feature            | Linear Regression      | Logistic Regression            |
| ------------------ | ---------------------- | ------------------------------ |
| **Main Use**       | Regression             | Classification                 |
| **Output**         | Continuous value       | Probability / Class            |
| **Output Range**   | Any real value         | 0 to 1 probability             |
| **Activation**     | Linear                 | Sigmoid                        |
| **Example**        | House Price Prediction | Spam Classification            |
| **Common Metrics** | MAE, MSE, RMSE, R²     | Precision, Recall, F1, ROC-AUC |

---

## 📊 Model Evaluation

The notebooks use several important classification metrics to evaluate model performance.

### 1. Accuracy

Accuracy measures the proportion of correctly classified observations.

```text
Accuracy = Correct Predictions / Total Predictions
```

Accuracy can sometimes be misleading when the dataset is highly imbalanced.

---

### 2. Precision

Precision measures how many observations predicted as positive were actually positive.

```text
Precision = TP / (TP + FP)
```

Where:

* **TP** = True Positive
* **FP** = False Positive

Precision is useful when false positives are important to control.

---

### 3. Recall

Recall measures how many actual positive observations were correctly identified.

```text
Recall = TP / (TP + FN)
```

Where:

* **TP** = True Positive
* **FN** = False Negative

Recall is useful when missing positive cases is costly.

---

### 4. F1-Score

F1-score is the harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

It provides a balance between **Precision** and **Recall**.

---

## 📈 ROC Curve

The **Receiver Operating Characteristic (ROC) Curve** evaluates the model across different classification thresholds.

It plots:

```text
True Positive Rate (TPR)
            vs
False Positive Rate (FPR)
```

### True Positive Rate

```text
TPR = TP / (TP + FN)
```

### False Positive Rate

```text
FPR = FP / (FP + TN)
```

The ROC curve helps analyze how well a model separates the two classes across different thresholds.

---

## 🏆 ROC-AUC Score

**ROC-AUC** represents the **Area Under the ROC Curve**.

The score generally ranges from:

```text
0 → Poor class separation
1 → Perfect class separation
```

A higher ROC-AUC indicates better class separation across classification thresholds.

---

## 📍 Decision Boundary

A **decision boundary** separates different predicted classes in the feature space.

For a two-dimensional dataset, the decision boundary can be visualized to understand how the model separates different classes.

```text
             Feature 2
                 ↑

        Class 1
       ●  ●  ●
     ●  ●  ●
   ●  ●  ●
---------------------------  Decision Boundary
          ○  ○  ○
        ○  ○  ○
      ○  ○  ○

                 → Feature 1
```

Decision boundary visualization is especially useful for understanding how a classification model makes predictions.

---

# ⚠️ Imbalanced Classification

Real-world classification datasets are often **imbalanced**.

An imbalanced dataset occurs when one class contains significantly more observations than another.

### Example

```text
Class 0 → 375 samples
Class 1 → 25 samples
```

Here:

* **Class 0** = Majority Class
* **Class 1** = Minority Class

---

## Why is Class Imbalance Important?

Consider a dataset containing:

```text
950 → Class 0
50  → Class 1
```

A model that predicts Class 0 for every observation could achieve:

```text
Accuracy = 95%
```

However, it would fail to identify any Class 1 observations.

Therefore, for imbalanced datasets, it is important to consider additional metrics such as:

* **Precision**
* **Recall**
* **F1-Score**
* **ROC-AUC**

---

## 🔽 Random UnderSampling

**Random UnderSampling** reduces the number of observations from the majority class.

### Example

```text
Before:

Class 0 → 375
Class 1 → 25


After UnderSampling:

Class 0 → 25
Class 1 → 25
```

The dataset becomes balanced by removing randomly selected observations from the majority class.

### Advantages

* **Reduces dataset size**
* **Can make training faster**
* **Balances class representation**

### Limitations

* **Potential loss of useful majority-class information**
* **Can reduce the amount of training data**
* Results may depend on which observations are removed

### Implementation

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)

X_resampled, y_resampled = rus.fit_resample(
    X_train,
    y_train
)
```

---

## 🔼 Random OverSampling

**Random OverSampling** increases the number of observations in the minority class by randomly duplicating existing minority-class observations.

### Example

```text
Before:

Class 0 → 375
Class 1 → 25


After OverSampling:

Class 0 → 375
Class 1 → 375
```

### Advantages

* **Does not remove majority-class observations**
* **Balances class representation**
* **Simple to implement**

### Limitations

* **Can increase the risk of overfitting**
* **Increases dataset size**
* Duplicated observations do not introduce new information

### Implementation

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

X_resampled, y_resampled = ros.fit_resample(
    X_train,
    y_train
)
```

---

# 📓 Notebook 01 — Logistic Regression Basics

## `Logistic_Regression_01_Basics.ipynb`

This notebook introduces the fundamental concepts of Logistic Regression and demonstrates how to build a basic classification model.

### Topics Covered

* **Introduction to Classification**
* **Logistic Regression**
* **Linear Regression vs Logistic Regression**
* **Sigmoid Function**
* **Probability Prediction**
* **Decision Threshold**
* **Dataset Preparation**
* **Train-Test Split**
* **Model Training**
* **Class Predictions**
* **Probability Predictions**
* **Classification Report**
* **Precision**
* **Recall**
* **F1-Score**
* **ROC Curve**
* **ROC-AUC Score**
* **Decision Boundary**
* **Model Visualization**

### Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Visualization
   ↓
Train-Test Split
   ↓
Logistic Regression
   ↓
Predictions
   ↓
Classification Report
   ↓
ROC Curve
   ↓
ROC-AUC Score
   ↓
Decision Boundary
```

---

# 📓 Notebook 02 — Logistic Regression with Imbalanced Data

## `Logistic_Regression_02_Imbalanced_Data.ipynb`

This notebook explores Logistic Regression when the dataset contains unequal class distributions.

### Topics Covered

* **Creating an Imbalanced Dataset**
* **Class Distribution**
* **Logistic Regression on Imbalanced Data**
* **Classification Report**
* **ROC-AUC Score**
* **ROC Curve**
* **Decision Boundary**
* **Random UnderSampling**
* **Model Training after UnderSampling**
* **Random OverSampling**
* **Model Training after OverSampling**
* **Comparison of Class Distributions**
* **Visualization of Resampled Datasets**

### Workflow

```text
Imbalanced Dataset
        │
        ├──────────────────┐
        │                  │
        ▼                  ▼
Original Logistic     Resampling
Regression                │
                          │
                   ┌──────┴──────┐
                   ▼             ▼
            UnderSampling   OverSampling
                   │             │
                   ▼             ▼
             Logistic       Logistic
             Regression     Regression
                   │             │
                   └──────┬──────┘
                          ▼
                  Model Evaluation
```

---

## 🛠️ Technologies & Libraries

| Category                    | Technologies     |
| --------------------------- | ---------------- |
| **Programming Language**    | Python           |
| **Data Manipulation**       | NumPy, Pandas    |
| **Data Visualization**      | Matplotlib       |
| **Machine Learning**        | Scikit-learn     |
| **Imbalanced Learning**     | Imbalanced-learn |
| **Development Environment** | Jupyter Notebook |

---

## 📦 Installation

Install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn jupyter
```

Or install dependencies from the main repository:

```bash
pip install -r ../requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Navigate to the Logistic Regression Folder

```bash
cd Machine-Learning-Notebooks/02_Logistic_Regression
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the Notebook

Start with:

```text
Logistic_Regression_01_Basics.ipynb
```

Then continue with:

```text
Logistic_Regression_02_Imbalanced_Data.ipynb
```

### 5. Run the Cells

Run the notebook cells sequentially and explore the generated:

* **Outputs**
* **Graphs**
* **Predictions**
* **Evaluation Metrics**
* **Decision Boundaries**

---

## 📁 Folder Structure

```text
02_Logistic_Regression/
│
├── Logistic_Regression_01_Basics.ipynb
├── Logistic_Regression_02_Imbalanced_Data.ipynb
└── README.md
```

---

## 📈 Visualizations Included

The notebooks include visualizations such as:

* **Class Distribution**
* **Scatter Plots**
* **ROC Curves**
* **Decision Boundaries**
* **UnderSampling Results**
* **OverSampling Results**
* **Resampled Dataset Distribution**

All notebook outputs and visualizations are preserved inside the `.ipynb` files so they can be viewed directly on GitHub.

---

## 🎯 Learning Objectives

After completing these notebooks, you should be able to:

* **Explain Logistic Regression**
* **Understand binary classification**
* **Explain the Sigmoid Function**
* **Interpret predicted probabilities**
* **Understand classification thresholds**
* **Train Logistic Regression using Scikit-learn**
* **Evaluate classification models**
* **Interpret Precision, Recall, and F1-Score**
* **Understand ROC Curves and ROC-AUC**
* **Visualize Decision Boundaries**
* **Identify Class Imbalance**
* **Apply Random UnderSampling**
* **Apply Random OverSampling**
* **Understand the trade-offs of different resampling techniques**

---

## 🌍 Real-World Applications

Logistic Regression can be applied to many classification problems, including:

* **Spam Detection**
* **Fraud Detection**
* **Customer Churn Prediction**
* **Medical Classification**
* **Credit Risk Classification**
* **Sentiment Classification**
* **Marketing Response Prediction**
* **Binary Outcome Prediction**

The appropriate model, features, threshold, and evaluation metrics depend on the specific application and the cost of different types of classification errors.

---

## 🔑 Key Concepts

| Concept                 | Description                                              |
| ----------------------- | -------------------------------------------------------- |
| **Supervised Learning** | Learning from labeled training data                      |
| **Classification**      | Predicting discrete class labels                         |
| **Sigmoid Function**    | Converts model output into a probability between 0 and 1 |
| **Decision Threshold**  | Converts probability into a class prediction             |
| **Precision**           | Measures correctness of positive predictions             |
| **Recall**              | Measures how many actual positives are identified        |
| **F1-Score**            | Harmonic mean of Precision and Recall                    |
| **ROC Curve**           | Shows TPR against FPR across thresholds                  |
| **ROC-AUC**             | Measures class separation across thresholds              |
| **Decision Boundary**   | Separates predicted classes in feature space             |
| **UnderSampling**       | Reduces majority-class observations                      |
| **OverSampling**        | Increases minority-class observations                    |

---

## 📌 Learning Path

```text
Logistic Regression
        ↓
Binary Classification
        ↓
Sigmoid Function
        ↓
Probability Prediction
        ↓
Decision Threshold
        ↓
Model Training
        ↓
Model Evaluation
        ↓
ROC Curve & ROC-AUC
        ↓
Decision Boundary
        ↓
Class Imbalance
        ↓
Random UnderSampling
        ↓
Random OverSampling
```

---

## 📂 Repository Structure

The main repository contains multiple machine learning algorithms and concepts.

```text
Machine-Learning-Notebooks/
│
├── 01_Linear_Regression/
│
├── 02_Logistic_Regression/
│   ├── Logistic_Regression_01_Basics.ipynb
│   ├── Logistic_Regression_02_Imbalanced_Data.ipynb
│   └── README.md
│
├── 03_K_Nearest_Neighbors/
│
├── 04_Decision_Tree/
│
├── 05_Random_Forest/
│
├── 06_SVM/
│
├── 07_Naive_Bayes/
│
├── 08_K_Means/
│
├── 09_Hierarchical_Clustering/
│
├── 10_PCA/
│
├── datasets/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 💡 Key Takeaways

* **Logistic Regression is primarily used for classification.**
* **The Sigmoid Function converts the model output into a probability.**
* **A decision threshold converts probability into a class label.**
* **Precision, Recall, and F1-Score provide more detailed evaluation than accuracy alone.**
* **ROC-AUC helps evaluate class separation across different thresholds.**
* **Decision boundaries help visualize how a classifier separates classes.**
* **Imbalanced datasets require careful evaluation and may benefit from resampling techniques.**
* **Random UnderSampling removes majority-class observations.**
* **Random OverSampling duplicates minority-class observations.**
