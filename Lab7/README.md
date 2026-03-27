# EEP 567 Lab 7: Ensemble Learning and Random Forest

## Title and Brief Description
This lab introduces ensemble learning techniques in machine learning, focusing on voting, bagging, boosting, stacking, and random forest methods. Students will apply these algorithms to the Kaggle credit card fraud detection dataset and evaluate their performance using standard metrics.

## What You'll Learn
- Fundamentals of ensemble learning
- Types of ensembles: voting, bagging, boosting, stacking, random forest
- How to preprocess and scale features
- Model evaluation metrics: accuracy, precision, recall, F1-score
- Practical implementation of ensemble methods using scikit-learn

## Quick Start
1. **Install Required Packages**:
   ```bash
   pip install pandas scikit-learn
   ```
2. **Dataset**: Ensure `creditcard.csv` is in the lab directory.
3. **Run the Notebooks**:
   - Open both the instructional and student notebooks in Jupyter or VS Code.
   - Complete the TODOs in the student notebook.

## ML Pipeline Overview
```
[Load Dataset] → [Feature Scaling] → [Train/Test Split] → [Model Training] → [Model Evaluation]
```
- Ensemble methods are applied after preprocessing and splitting.
- Evaluation metrics are computed for each model.

## Dataset
- **Source**: Kaggle Credit Card Fraud Detection
- **File**: `creditcard.csv`
- **Features**:
  - `Time`: Transaction time (scaled)
  - `Amount`: Transaction amount (scaled)
  - `V1`-`V28`: Principal components
  - `Class`: Fraud label (0 = normal, 1 = fraud)

## Expected Results
- Successful implementation of ensemble models
- Completion of all TODOs in the student notebook:
  - Voting ensemble: create and evaluate sub-classifiers
  - Bagging ensemble: test different configurations
  - Random forest: train and compare with single decision tree
- Model performance metrics displayed for each method

## Common Issues
- **Missing Dataset**: Ensure `creditcard.csv` is present
- **Package Errors**: Install all required packages with `pip install pandas scikit-learn`
- **Convergence Warnings**: Increase `max_iter` for logistic regression if needed
- **Import Errors**: Check that `lab_7_util.py` is in the directory
- **Kernel Issues**: Restart Jupyter kernel if code execution stalls

## References
- [Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning)
- [Ensemble Learning in Machine Learning](https://towardsdatascience.com/ensemble-learning-in-machine-learning-getting-started-4ed85eb38e00)
- [Random Forest](https://en.wikipedia.org/wiki/Random_forest)
- [Boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning))
- [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)
- [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
