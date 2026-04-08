# 📈 Linear Regression Project - Restaurant Profit Prediction

[![Python](https://img.shields.io/badge/Python-3.12.3-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.0-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.0-orange.svg)](https://matplotlib.org/)

## 📋 Project Overview

This project is part of the **Machine Learning Specialization** by DeepLearning.AI and Stanford University (Course 1: Supervised Machine Learning: Regression and Classification). The goal is to implement **linear regression with one variable** to predict restaurant franchise profits based on city population.

### 🎯 Problem Statement
As the CEO of a restaurant franchise, you want to expand your business to cities that may give higher profits. Using historical data from cities where restaurants are already established, you'll build a model to predict potential profits for new cities.

## 📊 Dataset

- **Feature (x_train)**: City population (in 10,000s)
- **Target (y_train)**: Monthly profit (in $10,000)
- **Training examples**: 97 cities

### Sample Data
| Population (10,000s) | Profit ($10,000) |
|---------------------|------------------|
| 6.1101              | 17.592           |
| 5.5277              | 9.1302           |
| 8.5186              | 13.662           |
| 7.0032              | 11.854           |
| 5.8598              | 6.8233           |

## 🧠 Model Implementation

### Linear Regression Model
```
f_wb(x) = wx + b
```

### Cost Function (Mean Squared Error)
```
J(w,b) = (1/2m) * Σ(f_wb(xⁱ) - yⁱ)²
```

### Gradient Descent Algorithm
```
w := w - α * (1/m) * Σ(f_wb(xⁱ) - yⁱ) * xⁱ
b := b - α * (1/m) * Σ(f_wb(xⁱ) - yⁱ)
```

## 📁 Repository Structure

```
DeepLearningAiProjects/
└── Machine-Learning-Specialization/
    └── Course1-Supervised-Machine-Learning-Regression-and-Classification/
        └── Linear_Regression_Project/
            ├── C1_W2_Linear_Regression.ipynb   # Main Jupyter notebook
            ├── utils.py                         # Helper functions
            ├── public_tests.py                  # Test functions
            ├── requirements.txt                 # Python dependencies
            └── README.md                        # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.12.3 or higher
- Git

### Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/MohamedAliZouariEng/DeepLearningAiProjects.git
cd DeepLearningAiProjects/Machine-Learning-Specialization/Course1-Supervised-Machine-Learning-Regression-and-Classification/Linear_Regression_Project
```

2. **Create and activate virtual environment**
```bash
# On Linux
python3 -m venv venv
source venv/bin/activate

```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### 📦 Dependencies
```
numpy>=1.26.0
matplotlib>=3.8.0
```

## 📈 Results

After running gradient descent with:
- **Learning rate (α)**: 0.01
- **Iterations**: 1500

The model found optimal parameters:
- **w (slope)**: 1.16636235
- **b (intercept)**: -3.63029144

### Sample Predictions
| Population | Predicted Profit |
|------------|-----------------|
| 35,000     | $4,519.77       |
| 70,000     | $45,342.45      |

### 📊 Visualization
The plot below shows the linear regression fit to the training data:

![Profit vs Population](https://github.com/MohamedAliZouariEng/DeepLearningAiProjects/blob/main/Machine-Learning-Specialization/Course1-Supervised-Machine-Learning-Regression-and-Classification/Linear_Regression_Project/images/output.png)

*Note: The plot will be generated when you run the notebook*

## 📝 Implementation Details

### Key Functions Implemented

1. **compute_cost()** - Calculates the cost J(w,b)
   - Iterates through all training examples
   - Computes prediction f_wb for each example
   - Sums squared errors and divides by 2m

2. **compute_gradient()** - Computes gradients for parameters
   - Calculates dj_dw and dj_db for each example
   - Averages gradients over all examples
   - Returns gradients for parameter updates

3. **gradient_descent()** - Performs batch gradient descent
   - Updates w and b simultaneously
   - Tracks cost history
   - Returns optimized parameters

## 📈 Model Performance

The model successfully learned the linear relationship between city population and restaurant profits, as evidenced by:
- Decreasing cost function over iterations
- Linear fit that follows the data trend
- Reasonable predictions for new populations

## 🔍 Key Learnings

- ✅ Implementing linear regression from scratch
- ✅ Understanding cost function and gradient descent
- ✅ Visualizing data and model predictions
- ✅ Making predictions with trained models
- ✅ Batch processing of training examples

## 📚 References

- [Machine Learning Specialization - DeepLearning.AI](https://www.deeplearning.ai/courses/machine-learning-specialization/)

## 🤝 Connect with Me

- **GitHub**: [MohamedAliZouariEng](https://github.com/MohamedAliZouariEng)
- **LinkedIn**: [Mohamed Ali Zouari](https://www.linkedin.com/in/mohamed-ali-zouari-eng)
