# DBSCAN Clustering

A practical and beginner-friendly collection of **Jupyter Notebooks** covering **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**, an unsupervised machine learning algorithm used for discovering dense groups of data points and identifying noise or outliers.

This section focuses on understanding the core concepts of density-based clustering, implementing DBSCAN using **Python and Scikit-learn**, understanding `eps` and `min_samples`, identifying core, border, and noise points, visualizing clusters, and understanding how DBSCAN differs from other clustering algorithms.

---

## 📚 Contents

| **#**  | **Notebook**                   | **Description**                                                                                                                                        |
| ------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **01** | [`DBSCAN.ipynb`](DBSCAN.ipynb) | Fundamentals of DBSCAN, density-based clustering, `eps`, `min_samples`, cluster formation, noise detection, cluster visualization, and interpretation. |

---

## 🧠 What is DBSCAN?

**DBSCAN** stands for **Density-Based Spatial Clustering of Applications with Noise**.

It is an **unsupervised machine learning algorithm** used for clustering data points based on the density of points in their surrounding neighborhood.

Unlike K-Means, DBSCAN does **not require the number of clusters to be specified beforehand**.

It can also identify points that do not belong to any sufficiently dense region and classify them as **noise or outliers**.

DBSCAN mainly uses two parameters:

```text
eps
min_samples
```

Where:

* **`eps`** → Defines the neighborhood radius around a data point.
* **`min_samples`** → Defines the minimum number of points required to form a dense region.

---

## 🔵 Core, Border and Noise Points

DBSCAN classifies data points into three main categories.

### Core Point

A point is considered a **Core Point** when its `eps` neighborhood contains at least the required number of points specified by `min_samples`.

```text
Core Point
    ↓
Enough nearby points
    ↓
Dense region
```

---

### Border Point

A **Border Point** is not dense enough to be a core point itself but lies within the neighborhood of a core point.

```text
Core Point
    ●
   / \
  ●   ●
       \
        ○
     Border Point
```

---

### Noise Point

A **Noise Point** does not belong to a sufficiently dense region and is not reachable from a core point.

In Scikit-learn, noise points are represented by:

```text
-1
```

For example:

```python
labels = db.fit_predict(X)

print(labels)
```

A result containing:

```text
-1
```

indicates a noise point.

---

## ⚙️ How DBSCAN Works

DBSCAN groups points based on their local density rather than trying to divide the dataset into a predefined number of clusters.

The general process is:

```text
Select a data point
       ↓
Find points within eps distance
       ↓
Check min_samples
       ↓
Is it a Core Point?
       ↓
   Yes       No
    ↓         ↓
Expand      Check whether
Cluster     it is a Border Point
    ↓
Identify Noise Points
    ↓
Final Clusters
```

The algorithm continues this process until all data points have been visited.

---

## 📏 `eps` Parameter

`eps` represents the maximum distance between two points for them to be considered neighbors.

Example:

```python
db = DBSCAN(eps=10)
```

Here:

```text
eps = 10
```

defines the neighborhood radius used by DBSCAN.

### Effect of `eps`

```text
Small eps
   ↓
Smaller neighborhoods
   ↓
More points may become noise

Large eps
   ↓
Larger neighborhoods
   ↓
More points may become connected
```

Choosing an appropriate value of `eps` is important because it directly affects cluster formation.

---

## 🔢 `min_samples` Parameter

`min_samples` specifies the minimum number of points required in an `eps` neighborhood for a point to be considered a core point.

Example:

```python
db = DBSCAN(
    eps=10,
    min_samples=2
)
```

Here:

```text
eps = 10
min_samples = 2
```

A point needs a sufficiently populated neighborhood according to this setting to act as a core point.

### Effect of `min_samples`

```text
Small min_samples
        ↓
Less density required
        ↓
More points can become Core Points

Large min_samples
        ↓
More density required
        ↓
Fewer regions qualify as Core Points
```

---

## 🎯 DBSCAN Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Visualization
   ↓
Choose eps
   ↓
Choose min_samples
   ↓
Find Dense Regions
   ↓
Identify Core Points
   ↓
Expand Clusters
   ↓
Identify Border Points
   ↓
Identify Noise
   ↓
Visualize Clusters
   ↓
Interpret Results
```

---

## 📊 Example Dataset

The notebook uses a simple two-dimensional dataset:

```python
X = np.array([
    [2, 3],
    [3, 4],
    [67, 43],
    [10, 4],
    [4, 5],
    [1, 1],
    [68, 44],
    [23, 54]
])
```

The data contains points distributed across different regions.

DBSCAN is applied using:

```python
db = DBSCAN(
    eps=10,
    min_samples=2
)

labels = db.fit_predict(X)
```

The resulting labels are then used to visualize the discovered clusters.

---

## 📈 Cluster Visualization

The notebook visualizes the DBSCAN clusters using Matplotlib.

```python
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

plt.title("DBSCAN Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Each cluster receives a different label.

Noise points are represented by:

```text
-1
```

and can be visualized separately using a different marker.

Example:

```python
for label in unique_labels:

    if label == -1:
        plt.scatter(
            X[labels == label, 0],
            X[labels == label, 1],
            label="Noise",
            marker="x"
        )

    else:
        plt.scatter(
            X[labels == label, 0],
            X[labels == label, 1],
            label=f"Cluster {label + 1}"
        )
```

---

## 🧪 Model Implementation

DBSCAN can be implemented using Scikit-learn:

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(
    eps=10,
    min_samples=2
)

labels = db.fit_predict(X)
```

The `fit_predict()` method performs clustering and directly returns the cluster labels.

---

## 🔍 Understanding Cluster Labels

The output of DBSCAN contains integer labels.

For example:

```text
[0, 0, 1, 0, 0, 0, 1, -1]
```

The labels can be interpreted as:

```text
0  → Cluster 1
1  → Cluster 2
-1 → Noise
```

The actual numeric cluster labels are identifiers assigned by the algorithm; their numeric values do not represent a ranking between clusters.

---

## 📉 DBSCAN vs K-Means

| **Feature**            | **DBSCAN**                   | **K-Means**                                          |
| ---------------------- | ---------------------------- | ---------------------------------------------------- |
| **Learning Type**      | Unsupervised                 | Unsupervised                                         |
| **Number of Clusters** | Not required beforehand      | Required beforehand                                  |
| **Main Principle**     | Density-based                | Centroid-based                                       |
| **Cluster Shape**      | Can detect irregular shapes  | Generally works best with roughly spherical clusters |
| **Noise Detection**    | Explicitly identifies noise  | Does not provide a dedicated noise label             |
| **Main Parameters**    | `eps`, `min_samples`         | `n_clusters`                                         |
| **Outlier Handling**   | Can classify points as noise | Sensitive to outliers                                |
| **Density-Based**      | Yes                          | No                                                   |

---

## 📈 Visualizations

The notebook includes visualizations such as:

* **Original Data Scatter Plot**
* **DBSCAN Cluster Visualization**
* **Cluster-wise Scatter Plot**
* **Noise / Outlier Visualization**
* **Colored Cluster Labels**

These visualizations help understand how DBSCAN groups points according to their density.

---

## 🛠️ Technologies & Libraries

| **Category**                | **Technologies** |
| --------------------------- | ---------------- |
| **Programming Language**    | Python           |
| **Numerical Computing**     | NumPy            |
| **Data Visualization**      | Matplotlib       |
| **Machine Learning**        | Scikit-learn     |
| **Development Environment** | Jupyter Notebook |

---

## 📦 Installation

Install the required libraries using:

```bash
pip install numpy matplotlib scikit-learn jupyter
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

### 2. Navigate to the DBSCAN Folder

```bash
cd Machine-Learning-Notebooks/15_DBSCAN
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the Notebook

Start with:

```text
DBSCAN.ipynb
```

### 5. Run the Cells

Run the notebook cells sequentially and explore the generated:

* **Cluster Labels**
* **Cluster Visualizations**
* **Noise Points**
* **Core / Border / Noise Concepts**
* **Density-Based Groups**

---

## 📁 Folder Structure

```text
15_DBSCAN/
│
├── DBSCAN.ipynb
└── README.md
```

---

## 📓 Notebook 01 — DBSCAN Clustering

### `DBSCAN.ipynb`

This notebook introduces the fundamental concepts of DBSCAN and demonstrates how density-based clustering can be implemented using Scikit-learn.

### Topics Covered

* **Introduction to Clustering**
* **Unsupervised Learning**
* **DBSCAN**
* **Density-Based Clustering**
* **`eps`**
* **`min_samples`**
* **Core Points**
* **Border Points**
* **Noise Points**
* **Cluster Labels**
* **Dataset Exploration**
* **Data Visualization**
* **DBSCAN Implementation**
* **Cluster Visualization**
* **Noise Detection**
* **DBSCAN vs K-Means**

### Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Visualization
   ↓
Choose eps & min_samples
   ↓
DBSCAN
   ↓
Find Dense Regions
   ↓
Create Clusters
   ↓
Identify Noise
   ↓
Visualize Results
   ↓
Interpret Clusters
```

---

## 🎯 Learning Objectives

After completing this notebook, you should be able to:

* **Explain DBSCAN**
* **Understand density-based clustering**
* **Understand `eps`**
* **Understand `min_samples`**
* **Differentiate Core, Border and Noise points**
* **Understand DBSCAN cluster labels**
* **Implement DBSCAN using Scikit-learn**
* **Visualize DBSCAN clusters**
* **Identify noise and outlier points**
* **Understand how parameter values affect clustering**
* **Compare DBSCAN with K-Means**
* **Understand when density-based clustering can be useful**

---

## 🌍 Real-World Applications

DBSCAN can be used for various clustering and anomaly-detection problems, including:

* **Geographical Location Clustering**
* **GPS Data Analysis**
* **Anomaly Detection**
* **Outlier Detection**
* **Customer Segmentation**
* **Image Processing**
* **Spatial Data Analysis**
* **Network Analysis**
* **Pattern Discovery**
* **Location-Based Services**

The suitability of DBSCAN depends on the structure of the dataset, feature representation, distance measure, and parameter selection.

---

## ⚠️ Important Considerations

### 1. Choosing `eps`

The value of `eps` strongly affects neighborhood formation and therefore cluster structure.

### 2. Choosing `min_samples`

Different values of `min_samples` can change which points are considered core points.

### 3. Feature Scaling

When features have very different scales, distance-based algorithms can be affected.

Feature scaling may therefore be important before applying DBSCAN.

### 4. Different Cluster Densities

Standard DBSCAN can have difficulty when meaningful clusters have substantially different densities.

### 5. High-Dimensional Data

Distance-based methods can become less effective as the number of dimensions increases, so dimensionality reduction or appropriate feature engineering may sometimes be useful.

---

## 🔑 Key Concepts

| **Concept**       | **Description**                                            |
| ----------------- | ---------------------------------------------------------- |
| **DBSCAN**        | Density-based unsupervised clustering algorithm            |
| **Density**       | Concentration of points within a region                    |
| **`eps`**         | Neighborhood radius                                        |
| **`min_samples`** | Minimum number of points required for a dense neighborhood |
| **Core Point**    | Point satisfying the density requirement                   |
| **Border Point**  | Point near a core point but not itself a core point        |
| **Noise**         | Point that does not belong to a cluster                    |
| **Cluster**       | Group of density-connected points                          |
| **`-1` Label**    | Noise label returned by Scikit-learn                       |

---

## 📌 Learning Path

```text
Clustering
     ↓
Unsupervised Learning
     ↓
Density-Based Clustering
     ↓
DBSCAN
     ↓
eps
     ↓
min_samples
     ↓
Core Points
     ↓
Border Points
     ↓
Noise Points
     ↓
Cluster Formation
     ↓
Visualization
     ↓
Parameter Analysis
     ↓
DBSCAN vs K-Means
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
│   ├── DBSCAN.ipynb
│   └── README.md
│
├── datasets/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 💡 Key Takeaways

* **DBSCAN is an unsupervised density-based clustering algorithm.**
* **It does not require the number of clusters to be specified beforehand.**
* **`eps` defines the neighborhood radius.**
* **`min_samples` defines the density requirement for core points.**
* **DBSCAN identifies Core, Border, and Noise points.**
* **Noise points are represented by `-1` in Scikit-learn.**
* **DBSCAN can discover clusters with irregular shapes.**
* **Parameter selection can significantly affect the resulting clusters.**
* **Feature scaling can be important when features have different numerical scales.**
* **DBSCAN can be useful for both clustering and identifying noise or outliers.**

---

## 🎓 Purpose of This Section

The goal of this section is to build a strong foundation in **density-based clustering** through practical implementation.

The focus is not only on using:

```python
DBSCAN()
```

but on understanding the complete clustering workflow:

```text
Understand the Data
       ↓
Explore the Data
       ↓
Visualize the Data
       ↓
Choose Parameters
       ↓
Apply DBSCAN
       ↓
Identify Clusters
       ↓
Detect Noise
       ↓
Visualize Results
       ↓
Interpret the Clustering
```

---

## 🔗 Part of the Machine Learning Journey

This DBSCAN section is part of the **Machine-Learning-Notebooks** repository, which contains practical implementations of machine learning algorithms using Python and Jupyter Notebooks.

> **Learn the concept → Implement it → Visualize it → Analyze it → Understand it**
