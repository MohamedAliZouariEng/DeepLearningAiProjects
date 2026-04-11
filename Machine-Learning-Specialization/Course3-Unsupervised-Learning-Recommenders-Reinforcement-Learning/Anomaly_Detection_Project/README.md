# 🔍 Anomaly Detection Project

> **Machine Learning Specialization - Course 3** | Unsupervised Learning, Recommenders, Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8+-orange.svg)](https://matplotlib.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red.svg)](https://jupyter.org/)

## 📋 Overview

This project implements an **anomaly detection algorithm** using a **Gaussian distribution model** to identify anomalous behavior in server computers. The algorithm analyzes server performance metrics to detect potential failures or irregular patterns in a network.

### 🎯 Key Features

- **Gaussian Distribution Modeling** - Fit parameters (μ, σ²) for each feature
- **Probability Threshold Selection** - Optimize ε using F1 score on validation set
- **Anomaly Detection** - Identify low-probability examples as anomalies
- **Multi-dimensional Analysis** - Handle datasets with 11+ features

## 📊 Dataset

The project works with two datasets:

| Dataset | Samples | Features | Description |
|---------|---------|----------|-------------|
| **2D Dataset** | 307 | 2 | Throughput (mb/s) & Latency (ms) |
| **High-Dimensional** | 1000 | 11 | Multiple server performance metrics |

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.12 or higher
pip (Python package manager)
```

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/MohamedAliZouariEng/DeepLearningAiProjects.git
cd DeepLearningAiProjects/Machine-Learning-Specialization/Course3-Unsupervised-Learning-Recommenders-Reinforcement-Learning/Anomaly_Detection_Project
```

2. **Create and activate virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```



## 🧠 Algorithm Implementation

### 1. Gaussian Parameter Estimation

```python
def estimate_gaussian(X):
    """Calculate mean and variance for each feature"""
    mu = 1/m * np.sum(X, axis=0)
    var = 1/m * np.sum((X - mu) ** 2, axis=0)
    return mu, var
```

### 2. Threshold Selection

```python
def select_threshold(y_val, p_val):
    """Find optimal epsilon using F1 score"""
    # Compute predictions, true/false positives/negatives
    # Calculate precision, recall, and F1 score
    # Return best epsilon and F1 score
```

### 3. Anomaly Detection

- Classify example as anomaly if: `p(x) < ε`
- Uses multivariate Gaussian distribution
- Optimizes threshold via cross-validation

## 📈 Results

### 2D Dataset Results
- **Best ε**: `8.99 × 10⁻⁵`
- **F1 Score**: `0.875`
- Anomalies detected and visualized with red circles

### High-Dimensional Dataset Results
- **Best ε**: `1.38 × 10⁻¹⁸`
- **F1 Score**: `0.615`
- **Anomalies Found**: `117` out of 1000 samples

## 📊 Visualizations

The notebook generates several visualizations:

| Figure | Description |
|--------|-------------|
| **Figure 1** | Scatter plot of training data (throughput vs. latency) |
| **Figure 2** | Gaussian distribution contours fitted to data |
| **Figure 3** | Anomaly detection results with outliers highlighted |

## 🔗 References

- [Machine Learning Specialization](https://learn.deeplearning.ai/specializations/machine-learning)
