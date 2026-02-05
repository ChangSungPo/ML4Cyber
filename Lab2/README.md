# Lab 2: ML Pipeline for Spam Detection

Build a spam email classifier using machine learning.

## What You'll Learn

| Topic | Description |
|-------|-------------|
| **Pre-processing** | Load data, remove duplicates, tokenize text |
| **Feature Extraction** | Convert text to numerical vectors (bag-of-words) |
| **Training** | Train a logistic regression classifier |
| **Evaluation** | Measure accuracy, precision, recall, F1 score |

## Quick Start

```bash
# Install dependencies
pip install nltk scikit-learn pandas numpy

# Extract dataset
tar -xf emails.tar.xz

# Open notebook
jupyter notebook lab-2-student-20250203.ipynb
```

## The 5 Exercises

| # | Task | Key Function |
|:-:|------|--------------|
| 1 | Remove duplicate emails | `df.drop_duplicates()` |
| 2 | Create bag-of-words features | `CountVectorizer().fit_transform()` |
| 3 | Train logistic regression | `classifier.fit()` |
| 4 | Print test predictions | `classifier.predict()` |
| 5 | Evaluate model metrics | `classification_report()`, `accuracy_score()` |

## ML Pipeline Overview

```
Raw Emails → Clean Text → Word Counts → Train Model → Predictions → Evaluate
     ↓           ↓            ↓             ↓            ↓           ↓
  emails.csv  tokenize    CountVec    LogisticReg   predict()   accuracy
```

## Key Concepts

### Bag-of-Words
Converts text to numbers by counting word occurrences:
- Each unique word = one column
- Each document = one row  
- Values = word counts

### Logistic Regression
Binary classifier that predicts probability of spam (0 or 1).

### Evaluation Metrics
- **Accuracy**: % correct predictions
- **Precision**: Of predicted spam, how many are actually spam?
- **Recall**: Of actual spam, how many did we catch?
- **F1 Score**: Balance of precision and recall

## Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: nltk` | `pip install nltk` |
| `LookupError: stopwords` | Run `nltk.download("stopwords")` |
| Memory error on `toarray()` | Don't convert sparse matrix - it's huge! |
| `NameError: classifier` | Complete TODO #3 first |

## Expected Results

Test set accuracy: **~98.6%**

## Files

| File | Description |
|------|-------------|
| `lab-2-student-20250203.ipynb` | Your working notebook |
| `emails.csv` | Dataset (5728 emails) |
| `lab-2-api-ref.pdf` | API reference guide |

## Need Help?

1. Check the demo cells before each TODO
2. Refer to `lab-2-api-ref.pdf` for function signatures
3. Ask your TA during lab sessions

