# ✍️ Neural Networks for Handwritten Digit Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)

## 📋 Overview

This project implements a **multiclass neural network** to recognize handwritten digits (0-9) using the **MNIST dataset subset**. The neural network is built using TensorFlow/Keras and achieves high accuracy in classifying handwritten digits.

## 🎯 Key Features

- ✍️ Handwritten digit recognition (0-9)
- 🧮 Implementation of **Softmax function** from scratch
- 🔢 **ReLU activation** in hidden layers
- 📊 Visualization of predictions and model performance
- 🎨 Interactive digit display and prediction

## 🏗️ Model Architecture

```
Input Layer (400 units) → Hidden Layer 1 (25 units, ReLU) → Hidden Layer 2 (15 units, ReLU) → Output Layer (10 units, Linear)
```

- **Input**: 20×20 pixel grayscale images flattened to 400-dimensional vectors
- **Hidden Layers**: Two dense layers with ReLU activation
- **Output Layer**: 10 units (one for each digit 0-9) with linear activation
- **Loss Function**: Sparse Categorical Crossentropy (with logits)
- **Optimizer**: Adam (learning rate = 0.001)


## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MohamedAliZouariEng/DeepLearningAiProjects.git
   cd DeepLearningAiProjects/Machine-Learning-Specialization/Course2-Advanced-Learning-Algorithms/Neural_Networks_For_Multiclass_Classification_Project
   ```

2. **Create and activate virtual environment**
   ```bash
   # On Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```


## 📊 Dataset

The dataset contains **5,000 training examples** of handwritten digits:
- Each image is **20×20 pixels** (grayscale)
- Images are flattened to **400-dimensional vectors**
- Labels range from **0 to 9** (10 classes)
- Subset of the famous **MNIST dataset**

## 🧪 Implementation Details

### 1. Softmax Function Implementation
```python
def my_softmax(z):
    a = np.exp(z) / np.sum(np.exp(z))
    return a
```

### 2. Neural Network Architecture
```python
model = Sequential([
    tf.keras.Input(shape=(400,)),
    Dense(units=25, activation="relu"),
    Dense(units=15, activation="relu"),
    Dense(units=10, activation="linear")
])
```

### 3. Model Training
- **Epochs**: 40
- **Batch Size**: 32 (default)
- **Loss Function**: SparseCategoricalCrossentropy with `from_logits=True`
- **Optimizer**: Adam (learning_rate=0.001)

## 📈 Results

- ✅ Successfully classifies handwritten digits with high accuracy
- 📉 Loss decreases consistently over training epochs
- 🎯 Predictions visualized alongside actual labels
- ⚡ Fast inference on single images

## 🔍 Key Concepts Covered

- **ReLU Activation**: Non-linear activation function that outputs max(0, z)
- **Softmax Function**: Converts logits to probability distribution
- **Multiclass Classification**: Handling 10 different digit classes
- **Neural Network Architecture**: Designing layers with appropriate dimensions
- **TensorFlow/Keras**: Building and training neural networks
- **Numerical Stability**: Using `from_logits=True` for better stability

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **NumPy** - Numerical computations
- **TensorFlow 2.0+** - Deep learning framework
- **Matplotlib** - Data visualization
- **Jupyter Notebook** - Interactive development

## 📚 Learning Outcomes

By completing this project, you'll understand:
- How to implement multiclass classification using neural networks
- The role of ReLU activation in hidden layers
- How to use Softmax for probability distributions
- Building neural networks with TensorFlow/Keras
- Processing image data for deep learning
- Evaluating model performance on classification tasks

## 🧪 Running Tests

The notebook includes built-in unit tests to verify:
- Softmax function implementation
- Model architecture correctness
- Tensor shapes and dimensions

## 📖 References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Original handwritten digit database
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Machine Learning Specialization](https://learn.deeplearning.ai/specializations/machine-learning/) - Course materials
