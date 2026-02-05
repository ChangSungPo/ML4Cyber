# Lab 5: Autoencoder and Clustering

Learn unsupervised machine learning with autoencoders for dimensionality reduction and clustering algorithms.

## What You'll Learn

| Topic | Description |
|-------|-------------|
| **Autoencoders** | Build neural networks to compress and reconstruct data |
| **Dimensionality Reduction** | Learn compact representations of high-dimensional features |
| **K-means Clustering** | Partition data into clusters based on similarity |
| **Gaussian Mixture Models** | Fit probability distributions to identify clusters |
| **Clustering Evaluation** | Measure clustering quality using entropy metrics |

## Quick Start

```bash
# Install dependencies
pip install torch matplotlib numpy scikit-learn

# Extract features from previous lab
tar -xf kddcup-1999-features.tar.xz

# Open notebook
jupyter notebook lab-5-student-20260128.ipynb
```

## The 5 Exercises

| # | Task | Key Concept |
|:-:|------|-------------|
| 1 | Build encoder model | `nn.Sequential`, `nn.Linear`, activation layers |
| 2 | Build decoder model | Mirror encoder architecture |
| 3 | Train enhanced autoencoder | Dropout, L2 regularization |
| 4 | Perform K-means clustering | `MiniBatchKMeans.fit_predict()` |
| 5 | Evaluate clustering metrics | MSE, entropy calculation |

## ML Pipeline Overview

```
Raw Features → Autoencoder → Dimensionality Reduction → Clustering → Evaluate
     ↓              ↓                    ↓                    ↓           ↓
  41 features  Encoder/Decoder     3D features        K-means/GM    Confusion Matrix
```

## Key Concepts

### Autoencoders
Neural networks that learn to compress data by:
1. **Encoder**: Compresses input to a bottleneck representation
2. **Decoder**: Reconstructs output from bottleneck
3. **Loss**: Minimize reconstruction error + regularization

Structure: Input → Hidden → Bottleneck → Hidden → Output

### K-means Clustering
- Groups samples into k clusters
- Minimizes within-cluster variance
- Uses Lloyd's algorithm (assign → update centroids)
- Best k found by analyzing entropy vs. cluster count

### Gaussian Mixture Models
- Fits k Gaussian distributions to data
- Uses EM algorithm (Expectation → Maximization)
- Probabilistic clustering vs. hard assignment

### Evaluation Metrics
- **MSE**: Average squared distance to cluster centroid (unsupervised)
- **Entropy**: Cluster purity based on true labels (semi-supervised)
- **Confusion Matrix**: Maps cluster assignments to true classes

## Dataset: KDD Cup 1999

Network intrusion detection dataset with:
- ~4.9 million records
- 41 features per connection
- Binary labels: Normal (0) or Attack (1)
- Attack types: DoS, R2L, U2R, Probing

Pre-processed features provided in `kddcup-1999-features.npz`:
- `kdd_feat_std`: StandardScaler-normalized features
- `kdd_feat_t_sne`: t-SNE transformed to 3D
- `y_std`: Binary labels

## Expected Results

- **Autoencoder**: Successfully reduces 41D features to 3D while preserving structure
- **K-means**: Entropy minimum at k=4, clear cluster separation
- **Gaussian Mixture**: Similar clustering patterns to K-means

## Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch` | `pip install torch` |
| `NotImplementedError` | Complete all TODO sections |
| Memory issues with large tensors | Use `torch.inference_mode()` for evaluation |
| GPU not found | CPU fallback automatic via `get_torch_device()` |

## Files

| File | Description |
|------|-------------|
| `lab-5-student-20260128.ipynb` | Your working notebook |
| `kddcup-1999-features.npz` | Pre-processed feature data |
| `README.md` | This guide |
| `TA_GUIDE.md` | Solutions (TAs only) |

## Need Help?

1. Check the demo cells before each exercise
2. Review PyTorch documentation: https://pytorch.org/docs/
3. Scikit-learn clustering: https://scikit-learn.org/stable/modules/clustering.html
4. Ask your TA during lab sessions

## References

1. KDD Cup 1999: https://kdd.ics.uci.edu/databases/kddcup99/
2. Autoencoders: https://en.wikipedia.org/wiki/Autoencoder
3. K-means: https://en.wikipedia.org/wiki/K-means_clustering
4. Gaussian Mixture: https://en.wikipedia.org/wiki/Mixture_model
5. EM Algorithm: https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
