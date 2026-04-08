# ✨ Advice for Applying Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)

> 📌 **Course:** Machine Learning Specialization - Course 2: Advanced Learning Algorithms  
> 🎓 **Instructor:** Andrew Ng  


## 🔍 Overview

This comprehensive practice lab explores essential techniques for **evaluating and improving machine learning models**. The project demonstrates how to diagnose and address common ML problems like **overfitting** and **underfitting** using systematic approaches.

### What You'll Learn
- ✅ How to properly split data into training, cross-validation, and test sets
- ✅ Methods to detect bias vs. variance problems
- ✅ Techniques for tuning model complexity and regularization
- ✅ How to build and evaluate neural network models
- ✅ Classification error calculation for categorical models

## 🎯 Learning Objectives

| Topic | Description |
|-------|-------------|
| **Data Splitting** | Learn to split datasets into training, validation, and test sets (60/20/20 split) |
| **Bias & Variance** | Identify underfitting vs. overfitting using learning curves |
| **Model Complexity** | Tune polynomial degree to find optimal model complexity |
| **Regularization** | Use cross-validation to select optimal λ (lambda) values |
| **Neural Networks** | Build simple and complex NN models for classification |
| **Error Analysis** | Calculate and interpret classification errors |


## 🚀 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MohamedAliZouariEng/DeepLearningAiProjects.git
cd DeepLearningAiProjects/Machine-Learning-Specialization/Course2-Advanced-Learning-Algorithms/Advice_for_Applying_Machine_Learning_Project
```

### 2️⃣ Create and Activate Virtual Environment

**Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 📚 Key Concepts Covered

### 🔹 Data Splitting Strategy
```
┌─────────────┬──────────────┬──────────────┐
│   Training  │   Validation │     Test     │
│     60%     │      20%     │     20%      │
└─────────────┴──────────────┴──────────────┘
```

### 🔹 Bias vs. Variance Diagnosis
| Symptom | Training Error | CV Error | Diagnosis |
|---------|---------------|----------|-----------|
| High Bias | High | High | Underfitting |
| High Variance | Low | High | Overfitting |
| Just Right | Low | Low | Good Fit |

### 🔹 Model Complexity Tuning
- Polynomial degree selection using validation curves
- Finding the "sweet spot" between underfitting and overfitting

### 🔹 Regularization (λ) Tuning
- Varying lambda values from 0 to 100
- Selecting optimal lambda based on CV performance

### 🔹 Neural Network Architectures
- **Complex Model:** 3 layers (120 → 40 → 6 units)
- **Simple Model:** 2 layers (6 → 6 units)

## ✏️ Exercises

This notebook contains **5 graded exercises**:

| Exercise | Topic | Description |
|----------|-------|-------------|
| **Ex 1** | `eval_mse()` | Calculate mean squared error for regression |
| **Ex 2** | `eval_cat_err()` | Calculate categorization error for classification |
| **Ex 3** | `model` | Build a complex 3-layer neural network |
| **Ex 4** | `model_s` | Build a simple 2-layer neural network |
| **Ex 5** | `model_r` | Compare model performance (analysis) |

## 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **NumPy** | Numerical computations |
| **scikit-learn** | Data splitting, linear regression, preprocessing |
| **TensorFlow/Keras** | Neural network implementation |
| **Matplotlib** | Data visualization and learning curves |
| **Jupyter** | Interactive development environment |

## 📖 References

- [Machine Learning Specialization - Coursera](https://learn.deeplearning.ai/specializations/machine-learning)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [TensorFlow Documentation](https://www.tensorflow.org/docs)

---
