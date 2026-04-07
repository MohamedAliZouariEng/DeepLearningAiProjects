# 🔢 Neural Networks for Binary Classification
### Handwritten Digit Recognition (0 vs 1)

This project is part of the **Deep Learning.AI Machine Learning Specialization**. It demonstrates the implementation of a neural network to perform binary classification on a subset of the MNIST dataset, specifically focusing on distinguishing between the handwritten digits **0** and **1**.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-013243?logo=numpy&logoColor=white)](https://numpy.org/)

---

## 📋 Project Overview
In this lab, a neural network is constructed and trained to recognize two handwritten digits. The project covers:
* **Data Representation:** Unrolling 20x20 pixel grayscale images into 400-dimensional vectors.
* **TensorFlow Implementation:** Building a `Sequential` model using Keras layers.
* **NumPy Implementation:** Manual coding of Forward Propagation and vectorized operations.


---

## 📂 Repository Structure
* **Path:** `DeepLearningAiProjects/Machine-Learning-Specialization/Course2-Advanced-Learning-Algorithms/Neural_Networks_For_Binary_Classification_Project/`
* **Main File:** `Neural_Networks_For_Binary_Classification_Project.ipynb` — The primary Jupyter Notebook containing the exercises and implementation.
* **Support Files:** `autils.py`, `public_tests.py`, and the dataset files.

---

## 🚀 Getting Started

Follow these steps to set up the project on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/MohamedAliZouariEng/DeepLearningAiProjects.git
cd DeepLearningAiProjects/Machine-Learning-Specialization/Course2-Advanced-Learning-Algorithms/Neural_Networks_For_Binary_Classification_Project/
```

### 2. Set Up Virtual Environment 🐍
It is highly recommended to use a virtual environment to manage dependencies.
```bash
# Create the environment
python3 -m venv venv

# Activate it
# On Linux:
source venv/bin/activate
```

### 3. Install Dependencies 📦
```bash
pip install -r requirements.txt
```
---

## 🛠️ Implementation Details

### The Dataset
* **Samples:** 1,000 training examples of digits 0 and 1.
* **Input (X):** 1000 x 400 matrix (20x20 pixels unrolled).
* **Output (y):** 1000 x 1 vector where `0` represents digit 0 and `1` represents digit 1.

### Model Architecture
The model typically consists of:
1.  **Input Layer:** 400 units.
2.  **Hidden Layers:** Dense layers with ReLU activation.
3.  **Output Layer:** 1 unit with a Sigmoid activation for binary probability.

---

## 📚 References
* **Specialization:** [Machine Learning Specialization by Andrew Ng](https://learn.deeplearning.ai/specializations/machine-learning/lesson/bw6i6/welcome-to-machine-learning!)

---
