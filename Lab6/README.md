# EEP 567 Lab 6: Detecting Credit Card Fraud with Decision Tree

## Brief Description

This lab introduces you to **machine learning-based fraud detection** using decision trees. You'll work with a real-world credit card transactions dataset, analyze its characteristics, handle class imbalance using SMOTE, and build decision tree classifiers to identify fraudulent transactions. The lab emphasizes the challenges of imbalanced datasets and why traditional accuracy metrics are insufficient for fraud detection.

---

## What You'll Learn

### Key Concepts
- **Class Imbalance**: Understanding the challenges of highly imbalanced datasets (0.172% fraud cases)
- **Data Preprocessing**: Feature scaling using `StandardScaler` and `RobustScaler` to handle outliers
- **SMOTE & Borderline SMOTE**: Techniques for over-sampling minority classes to balance training data
- **Decision Trees**: Tree-based classification model for pattern learning and prediction
- **Evaluation Metrics**: Precision, Recall, F1-Score, and Confusion Matrix for imbalanced classification
- **PCA Visualization**: Dimensionality reduction for feature exploration and visualization

### Learning Objectives
By the end of this lab, you will:
1. Analyze characteristics of credit card transaction features (time, amount, PCA-transformed features)
2. Identify temporal and value-based patterns distinguishing fraud from non-fraud transactions
3. Apply appropriate feature scaling techniques to raw transaction data
4. Implement data resampling strategies to address class imbalance
5. Train and evaluate multiple decision tree classifiers
6. Compare classifier performance using metrics suitable for imbalanced datasets

---

## Quick Start

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Basic knowledge of machine learning and pandas

### Installation

1. **Install required libraries** (run in the first code cell of the notebook):
   ```python
   %pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn
   ```

2. **Verify dataset availability**: Ensure `creditcard.csv` is in the lab directory

3. **Import utility module**: The lab uses `lab_6_util.py` for helper functions (timing and visualization)

### Running the Lab
- Open `lab-6-student-20260209.ipynb` in Jupyter
- Run cells sequentially from top to bottom
- Complete the TODO exercises as you encounter them
- Reference `lab-6-20260209.ipynb` for implementation guidance

---

## ML Pipeline Overview

```
Raw Dataset (creditcard.csv)
         ↓
[Feature Analysis]
  ├─ Opaque Features (V1-V28): PCA visualization
  ├─ Transaction Time: Temporal pattern analysis
  └─ Transaction Amount: Distribution and percentile analysis
         ↓
[Feature Scaling]
  ├─ Time: StandardScaler (after hour-of-day conversion)
  └─ Amount: RobustScaler (outlier-robust scaling)
         ↓
[Train-Test Split] (40% test size for adequate fraud samples)
         ↓
[Data Resampling] (Training set only)
  ├─ Original SMOTE
  └─ Borderline SMOTE
         ↓
[Decision Tree Training]
  ├─ DT (no resampling)
  ├─ DT with SMOTE
  └─ DT with Borderline SMOTE
         ↓
[Evaluation on Test Set]
  └─ Precision, Recall, F1-Score, Confusion Matrix
```

---

## Dataset

### Overview
- **Source**: Kaggle Credit Card Fraud Detection dataset
- **Size**: 284,807 transactions
- **Time Period**: 2-day collection window
- **Class Distribution**: Highly imbalanced (492 fraud cases, 284,315 non-fraud)
- **Fraud Rate**: 0.172%

### Features
| Feature Type | Count | Description |
|---|---|---|
| **Opaque Features** | 28 | V1-V28: PCA-transformed raw features (privacy-protected) |
| **Time** | 1 | Relative timestamp in seconds (already normalized by PCA) |
| **Amount** | 1 | Transaction amount in currency (raw, not normalized) |
| **Label** | 1 | Class: 0 (non-fraud) or 1 (fraud) |

### Key Characteristics
- **Opaque Features**: Already normalized (PCA units); separable in 3D space but require additional features for better performance
- **Time Pattern**: Most transactions occur during 16-hour daytime window; fraud shows increased proportion during night hours (1-8 and 24-32 hours)
- **Amount Pattern**: Long-tail distribution with outliers; fraud transactions average $122 vs. non-fraud $88; fraud concentrates in high-value tail (top 40%)

---

## Exercises and TODOs

### Exercise 1: Transaction Time Distribution Analysis
**Location**: Cell following "Now let's examine the distribution separately for fraud and non-fraud transactions..."

**Task**: Plot transaction time distribution for fraud and non-fraud transactions separately using histograms with different colors and transparency.

**Requirements**:
- Use `plt.hist()` for plotting
- Set different colors for fraud and non-fraud data
- Increase transparency (`alpha` parameter) for overlapping visibility
- Plot percentiles (not absolute amounts) using the `density=True` parameter
- Hint: Use the pre-defined `time_fraud` and `time_nonfraud` variables

---

### Exercise 2: Feature Scaling
**Location**: Cell in "Feature Scaling" section where you see `df_scaled = df.copy()`

**Task**: Scale transaction time and amount features using appropriate scalers.

**Requirements**:
1. Divide transaction time by 24 to convert to hour-of-day (modulo operation)
2. Apply `StandardScaler` to the normalized time and override the `Time` column
3. Apply `RobustScaler` to transaction amount and override the `Amount` column
4. Verify the first 5 scaled values match expected ranges

**Why these scalers?**
- `StandardScaler`: Appropriate for normalized, distribution-like data
- `RobustScaler`: Handles long-tail distributions with outliers using interquartile range (IQR)

**Formula**: $x_{\text{scaled}} = \frac{x - Q_{25}(x)}{Q_{75}(x) - Q_{25}(x)}$

---

### Exercise 3: Decision Tree Training
**Location**: Cell in "Decision Tree Classifier" section

**Task**: Train three decision tree classifiers with different resampling strategies.

**Requirements**:
1. **Train `dt`**: A basic `DecisionTreeClassifier` on original samples
2. **Train `dt_smote`**: Use `imblearn.pipeline.Pipeline` to apply SMOTE over-sampling before training a `DecisionTreeClassifier`
3. **Train `dt_borderline_smote`**: Use `imblearn.pipeline.Pipeline` to apply `BorderlineSMOTE` over-sampling before training a `DecisionTreeClassifier`

**Important Notes**:
- Use `imblearn.pipeline.Pipeline` (NOT `sklearn.pipeline.Pipeline`) because it supports resampling
- The pipeline automatically applies SMOTE only to the training data, not the test data
- Each pipeline should have two steps: the resampling method and the classifier
- Wrap training in `with lab_6_util.timeit("Training message")` to measure execution time

---

## Expected Results

After completing all exercises, you should observe:

### Evaluation Metrics Comparison

| Model | Expected Behavior |
|---|---|
| **DT (no resampling)** | High accuracy but low recall for fraud class; many fraud cases missed |
| **DT with SMOTE** | Over-sampled minority class; may show performance degradation compared to baseline |
| **DT with Borderline SMOTE** | More selective over-sampling at decision boundary; behavior between baseline and SMOTE |

### Key Findings
- Simple decision trees have limited effectiveness for this fraud detection task
- Over-sampling alone does not guarantee improved performance metrics (recall/F1-score)
- Ensemble methods (Random Forest) and more sophisticated techniques are needed for better results
- Recall and F1-score are more important than accuracy for imbalanced fraud detection

### Visualization Outputs
- 3D scatter plot of PCA-reduced opaque features showing fraud/non-fraud separation
- Transaction time distribution plots showing night-hour fraud concentration
- Transaction amount distribution showing long-tail pattern and fraud high-value concentration

---

## Common Issues & Troubleshooting

### 1. **ImportError: No module named 'seaborn' or 'imblearn'**
**Solution**: Run the pip install cell at the beginning of the notebook:
```python
%pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn
```
Ensure you're using the Jupyter cell's `%pip` magic command for correct environment.

---

### 2. **"lab_6_util module not found" error**
**Solution**: Verify `lab_6_util.py` exists in the same directory as the notebook. If missing, contact your instructor.

---

### 3. **NotImplementedError when running cells with TODO exercises**
**Solution**: This is expected! The student notebook has `raise NotImplementedError` placeholders. Replace these with your implementation following the exercise requirements above.

---

### 4. **SMOTE raises "ValueError: not enough samples in class..."**
**Solution**: This typically occurs if your training set is too small. Verify you're using the full dataset and the train-test split is properly configured with `test_size=0.4`.

---

### 5. **RobustScaler or StandardScaler requires 2D input**
**Solution**: Always reshape 1D arrays to 2D using `.values[:, None]` syntax:
```python
StandardScaler().fit_transform(data.values[:, None])
```

---

### 6. **Pipeline step naming issues**
**Solution**: When building pipelines, ensure each step has a valid name:
```python
Pipeline([
    ("step_name_1", transformer1),
    ("step_name_2", model)
])
```

---

### 7. **Metrics show all zeros or unexpected values**
**Solution**: 
- Verify predictions are made on the test set, not training set
- Check that `y_test` matches the shape and content of predictions
- Ensure the model was properly trained before evaluation

---

## References

1. [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. [Seaborn API Reference](https://seaborn.pydata.org/api.html)
3. [Scikit-learn RobustScaler Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
4. [Imbalanced-learn SMOTE Documentation](https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.over_sampling)
5. [Oversampling and Undersampling in Data Analysis](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis#SMOTE)
6. [SMOTE for Imbalanced Classification](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
7. [Decision Trees on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree)
8. [Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
9. [ROC Curves and Precision-Recall Curves for Imbalanced Classification](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/)
10. [Receiver Operating Characteristic (ROC) Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

---

## Additional Notes

- **Lab Progression**: This lab (Lab 6) introduces decision trees as a baseline. Lab 7 will cover Random Forests and ensemble methods for improved fraud detection performance.
- **Imbalanced Dataset Awareness**: Always evaluate imbalanced datasets using precision, recall, and F1-score rather than accuracy alone.
- **Feature Scaling Impact**: The choice of scaler significantly affects model performance, especially with outlier-rich data like transaction amounts.

