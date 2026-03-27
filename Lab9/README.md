# EEP 567 Lab 9: Instruction Set Architecture (ISA) Identification of Program Binaries

## Overview

This lab explores **machine learning techniques for binary classification**, focusing on identifying the Instruction Set Architecture (ISA) of compiled program binaries. You'll work with a real-world dataset of 50,000 Base64-encoded binaries from Praetorian's "Machine Learning Binaries" challenge and apply multiple feature extraction methods combined with classification algorithms.

**Lab Type:** Applied Machine Learning with Real Binary Data  
**Dataset Size:** 50,000 binaries  
**Classes:** 12 Instruction Set Architecture types  
**Expected Duration:** 2-3 hours

## What You'll Learn

By completing this lab, you'll gain experience with:

### Feature Engineering
- **Byte-Histogram and Endianness Features**: Building statistical features from raw binary data
- **Byte-level N-Gram TF-IDF**: Text vectorization techniques applied to byte sequences
- **Hex-level N-Gram TF-IDF**: Feature extraction from hexadecimal representations
- **Dimensionality Reduction**: Using Kernel PCA for high-dimensional sparse features

### Classification Models
- Linear Support Vector Machines (SVM)
- Logistic Regression
- Decision Tree Classifier
- Random Forest

### Evaluation Metrics
- Accuracy, Precision, Recall, and F1-Score
- Confusion Matrices for detailed performance analysis
- Train/Test data evaluation for assessing generalization

### Data Processing Concepts
- Base64 decoding
- Data deduplication and balancing
- Stratified train/test splitting
- Sparse matrix operations

## Quick Start

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- 4GB+ RAM (for processing 50k samples)

### Installation

1. Install required packages:
```bash
pip install numpy scipy scikit-learn termcolor
```

2. Verify the data files are present:
   - `binaries-dataset/Base64EcncodedBinaries-50k.txt`
   - `binaries-dataset/LabelsOfBinaries-50k.txt`

3. Open the student notebook:
```bash
jupyter notebook lab-9-student-20260222.ipynb
```

### Running the Lab

Execute cells sequentially from top to bottom. Each cell builds on previous results. Key execution checkpoints:

| Cell | Purpose | Status Check |
|------|---------|--------------|
| 1-2 | Setup & dependencies | No errors in pip install |
| 3-5 | Data loading & preprocessing | Statistics printed with 50k total samples |
| 6-8 | Byte-histogram features | `feats_byte_hist_endian` shape: (50k, 260) |
| 9-11 | Byte-level TF-IDF | `feats_tf_idf_123` shape: (50k, 32k+) |
| 12-13 | Dimensionality reduction | `feats_tf_idf_123_kpca` shape: (50k, 300) |
| 14-16 | Hex-level TF-IDF | `feats_tf_idf_hex_123` ready for evaluation |
| 17+ | Model training & evaluation | Classification reports printed |

## ML Pipeline Overview

```
Raw Binary Data (Base64-encoded)
        ↓
    Decode to Bytes
        ↓
    ┌───────────────────────────────────────────┐
    │        Feature Extraction Methods         │
    ├───────────────────────────────────────────┤
    │ 1. Byte-Histogram + Endianness (260 dims) │
    │ 2. Byte-Level TF-IDF (32k dims → PCA 300)│
    │ 3. Hex-Level TF-IDF (variable dims)      │
    └───────────────────────────────────────────┘
        ↓
    Train/Test Split (20%/5% stratified)
        ↓
    ┌───────────────────────────────────────────┐
    │      Classification Models (4 options)    │
    ├───────────────────────────────────────────┤
    │ • Linear SVM                              │
    │ • Logistic Regression                     │
    │ • Decision Tree                           │
    │ • Random Forest                           │
    └───────────────────────────────────────────┘
        ↓
    Evaluate (Accuracy, Precision, Recall, F1)
        ↓
    Compare Feature/Model Performance
```

## Dataset

### Composition
- **Total Samples:** 50,000 unique binaries (after deduplication)
- **Duplicates:** ~100-200 (removed during preprocessing)
- **Classes:** 12 ISA types (balanced distribution)

### Architecture Types
```
avr, alphaev56, arm, m68k, mips, mipsel, 
powerpc, s390, sh4, sparc, x86_64, xtensa
```

### Data Format
- **Input:** Base64-encoded binary strings (~88 characters = ~66 bytes)
- **Binary Size Range:** 60-70 bytes on average
- **Labels:** String labels (architecture name) → mapped to integers (0-11)

### Data Distribution
The dataset is **well-balanced**: each ISA class contains approximately 4,000-4,200 samples. No under/oversampling is required.

## Exercises & TODOs

The student notebook contains **3 main TODO sections** requiring implementation:

### TODO 1: Data Loading and Visualization
**Location:** Data Pre-processing cell  
**Tasks:**
- Decode Base64-encoded strings into byte objects
- Convert bytes to hexadecimal string representation
- Format hex strings with byte-level spacing (groups of 2 hex chars separated by spaces)

**Hints:**
- Python's `base64` module provides decoding functions
- Use the `.hex()` method on byte objects
- String slicing and join operations are useful for formatting

### TODO 2: Byte-Level TF-IDF Features
**Location:** Byte-level 1,2,3-Gram TF-IDF Features cell  
**Tasks:**
- Fit a TfidfVectorizer for unigrams and bigrams on byte sequences
- Fit a separate TfidfVectorizer for trigrams with feature limit
- Combine both feature matrices using sparse matrix concatenation

**Hints:**
- Use `TfidfVectorizer` with `analyzer="char"` and `encoding="latin1"`
- Set appropriate `ngram_range` parameters
- `scipy.sparse.hstack` merges sparse matrices horizontally

### TODO 3: Hex-Level TF-IDF Features
**Location:** Hex-level (4-bit) 1,2,3-Gram TF-IDF Features cell  
**Tasks:**
- Convert all binaries to hexadecimal representation
- Fit TfidfVectorizer on hex characters with 1-, 2-, and 3-grams
- Store the transformed features

**Hints:**
- No encoding parameter needed (hex is standard ASCII)
- Treat hex strings as regular text for TfidfVectorizer
- All characters in hex strings are 0-9, a-f

## Expected Results

### Performance Benchmarks

The four classification models should achieve different performance levels across feature types:

**Byte-Histogram Features (~260 dimensions):**
- **Fastest** to train (seconds)
- **Lowest** accuracy (~60-75%)
- Good baseline for comparison

**Byte-Level TF-IDF + KernelPCA (~300 dimensions):**
- **Moderate** training time
- **High** accuracy (~85-95%)
- Better generalization with dimensionality reduction

**Hex-Level TF-IDF (variable dimensions):**
- **Slower** training (high-dimensional)
- **Highest** accuracy (~90-98%)
- Captures fine-grained patterns but computationally expensive

### Model Comparison
Across all feature types:
- **Random Forest & SVM** typically outperform Decision Tree
- **Logistic Regression** provides good baseline performance
- Training/test accuracy gap indicates generalization capability

### Evaluation Output
Each model produces:
```
Confusion Matrix: 12x12 matrix showing classification per ISA type
Classification Report:
  - Precision per class (how accurate predictions are)
  - Recall per class (how many of each class are found)
  - F1-Score per class (harmonic mean of precision/recall)
  - Macro/weighted averages across all classes
```

## Important Concepts

### Feature Extraction Methods

**Byte-Histogram:**
- Counts occurrence of each byte value (0-255): 256 features
- Detects endianness patterns (4 specific 2-byte sequences): 4 features
- Total: 260-dimensional normalized vector
- ✓ Fast, interpretable | ✗ Loses sequential information

**Byte-Level TF-IDF:**
- Treats each binary as text where "characters" are bytes
- Generates unigrams (1 byte), bigrams (2 bytes), trigrams (3 bytes)
- TF-IDF scoring emphasizes discriminative byte patterns
- Trigrams capped at 10,000 features to manage dimensionality
- Total: ~32,000+ dimensions (reduced to 300 via KernelPCA)
- ✓ Captures sequential patterns | ✗ High dimensionality, slower

**Hex-Level TF-IDF:**
- Represents binaries as hexadecimal strings (doubles length)
- Applies same TF-IDF n-gram approach to hex characters
- Finer granularity (4-bit patterns) vs byte-level (8-bit)
- ✓ Fine-grained features | ✗ Even higher dimensionality

### Data Preprocessing Steps
1. **Loading:** Read Base64 strings and labels from files
2. **Decoding:** Convert Base64 → bytes
3. **Deduplication:** Remove duplicate binaries
4. **Label Mapping:** String labels → integer indices (0-11)
5. **Train/Test Split:** Stratified split preserves class distribution

---

## Common Issues & Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` for dataset files | Data path incorrect or files missing | Ensure `binaries-dataset/` folder exists with both `.txt` files |
| `NotImplementedError` errors | TODO sections not completed | Complete all TODO implementations before running dependent cells |
| `MemoryError` during TF-IDF | Processing full 50k sparse matrix | Use the KernelPCA cell to reduce dimensions; check available RAM |
| SVM/Logistic Regression slow | Convergence issues with high dims | KernelPCA reduction helps; increase `max_iter` parameter if needed |
| `ImportError` for packages | Dependencies not installed | Run `pip install numpy scipy scikit-learn termcolor` again |
| KernelPCA timeout | RBF kernel on large sparse matrix | Normal for first run; may take 5-10 minutes. Use `await_terminal` |
| Confusion matrix hard to read | Output formatting | Use `print(confusion_matrix(...))` to see cleaner format |
| Accuracy < 50% for all models | Feature extraction issue | Verify TODOs are correctly implemented; check feature shapes |

### Performance Optimization

If experiencing slowdowns:

1. **Reduce dataset size** (for testing):
   ```python
   # Take first 10k samples instead of 50k
   binaries = binaries[:10000]
   labels = labels[:10000]
   ```

2. **Skip hex-level features** for now:
   - Most computationally expensive
   - Test byte-histogram and byte-level TF-IDF first

3. **Adjust KernelPCA parameters**:
   - Reduce `KERNEL_PCA_DIMS` from 300 to 200 for faster computation
   - Use linear kernel for faster processing (less accurate)

4. **Parallel processing**:
   - Random Forest uses all cores by default
   - Use `n_jobs=-1` for Logistic Regression/SVM if available

## References

[1] "Tech challenge: Machine learning binaries," Praetorian, Feb 2021.  
     Available: https://www.praetorian.com/challenges/machine-learning-challenge/#how-to-play

[2] J. Clemens, "Automatic classification of object code using machine learning," Digital Investigation, vol. 14, pp. S156–S162, 2015.

### Additional Resources

**scikit-learn Documentation:**
- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

**Background Reading:**
- Feature extraction from binary data (see reference [2])
- TF-IDF in text analysis and its application to binary data
- Kernel methods and dimensionality reduction

## File Structure

```
lab-9/
├── README.md                                    ← You are here
├── lab-9-20260222.ipynb                        (Full instructional notebook)
├── lab-9-student-20260222.ipynb               (TODO-based student notebook)
├── lab-9-20250304.ipynb                       (Older version)
├── lab-9-student-20250304.ipynb               (Older version)
└── binaries-dataset/
    ├── Base64EcncodedBinaries-50k.txt         (50k binaries in Base64)
    └── LabelsOfBinaries-50k.txt               (50k corresponding labels)
```

## Tips for Success

✓ **Start with byte-histogram features** — they're the fastest and easiest to understand  
✓ **Understand the data first** — run the data preprocessing cell and review the statistics  
✓ **Compare feature methods systematically** — note which features work best for each model  
✓ **Watch for overfitting** — compare training vs test accuracy to identify generalization issues  
✓ **Use the timing framework** — `timeit` context manager helps identify bottlenecks  
✓ **Review confusion matrices** — they reveal which ISA types are confused with each other  
✓ **Experiment safely** — the full instructional notebook shows complete solutions for reference

## Getting Help

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Review the **full instructional notebook** (lab-9-20260222.ipynb) for reference implementations
3. Verify all **dependencies are installed** and notebook kernel is active
4. Check that **data files exist** and paths are correct
5. Ensure **all cells run sequentially** without skipping any