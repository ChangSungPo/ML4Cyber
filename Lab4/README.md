# Lab 4: Dimensionality Reduction and Visualization

## Overview

This lab introduces **unsupervised machine learning** techniques for dimensionality reduction. You'll learn to transform high-dimensional data into lower dimensions for visualization and analysis, applying these methods to network intrusion detection.

## Learning Objectives

By completing this lab, you will:
- Understand why feature scaling is essential for ML algorithms
- Compare MinMaxScaler vs StandardScaler and when to use each
- Apply Principal Component Analysis (PCA) for linear dimensionality reduction
- Use Kernel PCA to handle non-linear relationships
- Implement t-SNE for high-quality visualizations
- Interpret and compare results from different techniques

## Dataset

**KDD Cup 1999 Network Intrusion Dataset**
- ~490,000 network connection records (10% subset)
- 41 features per connection (numerical and categorical)
- Labels: "normal" or specific attack type

Attack categories:
| Type | Description |
|------|-------------|
| DoS | Denial-of-service (e.g., SYN flood) |
| R2L | Remote-to-local access (e.g., password guessing) |
| U2R | Privilege escalation (e.g., buffer overflow) |
| Probing | Network scanning (e.g., port scan) |

## Prerequisites

### Required Libraries
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Data Setup
Extract the dataset before running the notebook:
```bash
tar -xf kddcup.tar.xz
```

## Lab Structure

### Part 1: Data Loading & Exploration
- Load the KDD Cup dataset
- Examine feature distributions and attack types

### Part 2: Feature Scaling
- **MinMaxScaler**: Rescale to [0, 1] range
- **StandardScaler**: Transform to zero mean, unit variance
- **Outlier removal**: Handle extreme values that skew visualizations

### Part 3: Dimensionality Reduction

**PCA (Principal Component Analysis)**
- Linear technique based on SVD
- Finds directions of maximum variance
- Fast but limited to linear relationships

**Kernel PCA**
- Non-linear extension using kernel trick
- Try different kernels: polynomial, RBF, sigmoid
- Can reveal complex data structures

**t-SNE**
- Non-linear technique for visualization
- Preserves local neighborhood structure
- Perplexity parameter controls neighborhood size

### Part 4: Visualization & Interpretation
- 2D and 3D scatter plots
- Compare separation of normal vs attack traffic
- Analyze which technique works best

## Key Concepts

### Why Dimensionality Reduction?
1. **Visualization**: Humans can only perceive 2-3 dimensions
2. **Noise reduction**: Remove irrelevant features
3. **Computational efficiency**: Fewer features = faster algorithms
4. **Curse of dimensionality**: Many algorithms struggle in high dimensions

### Feature Scaling Comparison
| Scaler | Formula | Use When |
|--------|---------|----------|
| MinMax | (x - min)/(max - min) | Bounded features needed |
| Standard | (x - μ)/σ | Gaussian-like distributions |

### Choosing a Technique
| Technique | Linear? | Speed | Best For |
|-----------|---------|-------|----------|
| PCA | Yes | Fast | Initial exploration |
| Kernel PCA | No | Medium | Non-linear separable data |
| t-SNE | No | Slow | High-quality visualization |

## Tips for Success

1. **Always scale your data** before applying dimensionality reduction
2. **Remove outliers** when using StandardScaler for visualization
3. **Try multiple techniques** - different methods reveal different patterns
4. **Experiment with parameters** - kernel type, perplexity, etc.
5. **Interpret carefully** - t-SNE distances are not directly meaningful

## Submission

Submit your completed notebook: `lab-4-student-20260128.ipynb`

Ensure all code cells execute without errors.

## Resources

- [sklearn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [sklearn Decomposition Guide](https://scikit-learn.org/stable/modules/decomposition.html)
- [t-SNE Explained (StatQuest)](https://www.youtube.com/watch?v=NEaUSP4YerM)
- [Understanding PCA](https://lazyprogrammer.me/tutorial-principal-components-analysis-pca/)
who is this claude?

