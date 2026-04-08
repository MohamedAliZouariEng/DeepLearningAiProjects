# 🍄 Decision Trees Project - Mushroom Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-orange.svg)](https://matplotlib.org/)

## 📋 Project Overview

This project implements a **Decision Tree classifier from scratch** to determine whether mushrooms are **edible** 🍽️ or **poisonous** ☠️ based on their physical attributes. This is part of the DeepLearning.AI Machine Learning Specialization - Course 2: Advanced Learning Algorithms.

### 🎯 Problem Statement

> You are starting a company that grows and sells wild mushrooms. Since not all mushrooms are edible, you need to identify which mushrooms can be sold safely based on their physical characteristics.

## 📊 Dataset

The dataset contains **10 mushroom examples** with three features:

| Feature | Description | Values |
|---------|-------------|--------|
| Cap Color | Color of the mushroom cap | Brown (1) / Red (0) |
| Stalk Shape | Shape of the mushroom stalk | Tapering (1) / Enlarging (0) |
| Solitary | Whether the mushroom grows alone | Yes (1) / No (0) |

**Target Variable:**
- `1` = Edible ✅
- `0` = Poisonous ❌

## 🏗️ Implementation Details

### Core Functions Implemented

| Function | Description |
|----------|-------------|
| `compute_entropy()` | Calculates impurity at a node using entropy formula |
| `split_dataset()` | Splits data into left/right branches based on a feature |
| `compute_information_gain()` | Measures information gain from splitting on a feature |
| `get_best_split()` | Identifies the best feature for splitting |

### 📐 Mathematical Foundation

**Entropy Formula:**
```
H(p₁) = -p₁·log₂(p₁) - (1-p₁)·log₂(1-p₁)
```

**Information Gain:**
```
IG = H(node) - (w_left·H(left) + w_right·H(right))
```

### 🌳 Decision Tree Structure

```
Depth 0, Root: Split on feature: 2 (Solitary)
├── Depth 1, Left: Split on feature: 0 (Brown Cap)
│   ├── Left leaf node: [0, 1, 4, 7] → Edible
│   └── Right leaf node: [5] → Poisonous
└── Depth 1, Right: Split on feature: 1 (Tapering Stalk Shape)
    ├── Left leaf node: [8] → Edible
    └── Right leaf node: [2, 3, 6, 9] → Poisonous
```

## 📈 Results

Information Gain at Root Node:
- **Brown Cap**: 0.0349
- **Tapering Stalk Shape**: 0.1245  
- **Solitary**: 0.2781 ⭐ **(Best Split)**

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/MohamedAliZouariEng/DeepLearningAiProjects.git

# Navigate to the project directory
cd DeepLearningAiProjects/Machine-Learning-Specialization/Course2-Advanced-Learning-Algorithms/Decision_trees_Project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

## 🧪 Testing

The implementation passes all unit tests:

```bash
✓ compute_entropy_test passed
✓ split_dataset_test passed  
✓ compute_information_gain_test passed
✓ get_best_split_test passed
```

## 📚 Key Learning Outcomes

- ✅ Understanding of Decision Tree algorithms
- ✅ Implementation of entropy and information gain calculations
- ✅ Building recursive tree structures
- ✅ Handling categorical features with one-hot encoding
- ✅ Visualizing decision boundaries


## 📁 Repository Structure

```
Decision_trees_Project/
├── Decision_trees_project.ipynb   # Main Jupyter notebook
├── requirements.txt                # Python dependencies
├── utils.py                       # Helper functions
├── public_tests.py               # Unit tests
├── images/                       # Visualization assets
└── README.md                     # This file
```

## 🔗 References

- [DeepLearning.AI Machine Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/)
- [Decision Trees - Scikit-learn Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [Information Gain in Decision Trees](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)
