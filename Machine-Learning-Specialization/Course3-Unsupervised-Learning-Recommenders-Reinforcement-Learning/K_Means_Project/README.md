#  🖼️K-Means Clustering for Image Compression

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24.0-blue.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.0-blue.svg)](https://matplotlib.org/)

A comprehensive implementation of the K-means clustering algorithm from scratch, applied to image compression. This project demonstrates unsupervised learning techniques to reduce the number of colors in an image while preserving its visual quality.

## 🎯 Overview

This project implements the K-means clustering algorithm for two primary purposes:

1. **Learning Phase**: Understanding K-means on a synthetic 2D dataset to visualize how clusters form iteratively
2. **Application Phase**: Using K-means for image compression by reducing the number of colors from thousands to just 16 representative colors

The image compression achieves approximately **6x compression** by storing only 16 colors and their pixel assignments instead of full RGB values for each pixel.

## ✨ Features

- **From-Scratch Implementation**: Complete implementation of K-means algorithm without using high-level ML libraries
- **Interactive Visualization**: Real-time visualization of centroid movement during clustering
- **Image Compression**: Apply K-means to compress images by reducing color palette
- **Random Initialization**: Multiple random initializations to find optimal clustering
- **Performance Metrics**: Visual comparison between original and compressed images

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/MohamedAliZouariEng/DeepLearningAiProjects.git

# Navigate to the project directory
cd DeepLearningAiProjects/Machine-Learning-Specialization/Course3-Unsupervised-Learning-Recommenders-Reinforcement-Learning/K_means_Clustering

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```


## 🔧 How It Works

### K-means Algorithm
The algorithm follows an iterative two-step process:

1. **Assignment Step**: Assign each data point to the closest centroid
2. **Update Step**: Recompute centroids as the mean of assigned points

```python
# Pseudocode
initialize centroids randomly
for iteration in range(max_iters):
    idx = find_closest_centroids(X, centroids)
    centroids = compute_centroids(X, idx, K)
```

### Image Compression Workflow

1. **Load Image**: Read image as RGB pixel matrix (128×128×3)
2. **Reshape**: Convert to 2D array of pixels (16384×3)
3. **Apply K-means**: Find 16 representative colors (centroids)
4. **Compress**: Replace each pixel with its closest centroid color
5. **Reconstruct**: Reshape back to original dimensions

## 📊 Results

### Sample Dataset Clustering
The algorithm successfully identifies natural clusters in 2D synthetic data, with centroids moving to optimal positions through iterations.

### Image Compression Results

| Metric | Original | Compressed (K=16) |
|--------|----------|-------------------|
| Colors | ~16,000  | 16                |
| Bits   | 393,216  | 65,920            |
| Compression Ratio | 1x | ~6x |

**Visual Comparison**:
- Original image (128×128 pixels, full color range)
- Compressed image (using only 16 colors, preserving visual characteristics)

## 🚀 Usage

### Key Parameters to Experiment With

```python
# Number of clusters (colors)
K = 16  # Try values: 8, 16, 32, 64

# Number of iterations
max_iters = 10  # Typically converges within 10 iterations

# Random initialization
initial_centroids = kMeans_init_centroids(X, K)
```

## 📈 Technical Details

### Algorithm Complexity
- **Time Complexity**: O(I × K × m × n)
  - I = number of iterations
  - K = number of clusters
  - m = number of data points
  - n = number of features

### Implementation Highlights
- Vectorized operations using NumPy for efficiency
- Euclidean distance calculation for nearest centroid
- Mean computation for centroid updates

## 📚 References

- [Andrew Ng's Machine Learning Course](https://www.deeplearning.ai/courses/machine-learning-specialization/)

