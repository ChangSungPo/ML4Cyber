# Malware Detection with PE File Analysis

##  Project Overview
This project is part of the **Machine Learning for Cybersecurity** course. We are developing automated, data-driven countermeasures to identify and classify malware variants into their respective families.

##  Group Members
* **Sung-Po Chang**
* **Mike Huang** 
* **Bryan Gonzalez** 

##  Problem Statement
We are addressing three primary technical hurdles:
1. **Massive Data Scale**: Managing a dataset exceeding 500GB uncompressed.
2. **Feature Representation**: Converting raw byte sequences (`.bytes`) and disassembly files (`.asm`) into numerical representations.
3. **Class Imbalance**: Addressing the "long-tail" distribution where certain malware families have significantly fewer samples.

##  The Dataset
We are utilizing the **Microsoft Malware Classification Challenge (BIG 2015)** dataset.
* **Format**: `.bytes` (hex dumps) and `.asm` (disassembled metadata).
* **Classes**: 9 distinct malware families.

##  Proposed Methodology
We will evaluate several machine learning architectures:
* **Baselines**: Logistic Regression, Naive Bayes, and KNN.
* **Ensembles**: Random Forest, XGBoost, and LightGBM.

##  Project Timeline
* **Week 1**: Environment setup and repository initialization.
* **Weeks 2-3**: EDA and data scaling strategies.
* **Week 4**: Model selection and functional testing.
* **Weeks 5-6**: Model training and comparison.
* **Week 7**: Final results and presentation.

## Model Evolution:
* Baseline Model: Initial Random Forest achieved 92.20% accuracy but struggled with confusion between Class 4 and Class 6.
* Metadata-Enhanced Model: Resolved the "obfuscation blindspot" by adding structural metadata, increasing accuracy to 94.04% and significantly improving F1-scores for Class 4 and Class 7.
* Final Ensemble Models: Scaled to the full dataset using Random Forest and XGBoost, achieving a final accuracy of 99%.

## Final Findings & Analysis
* Metadata Utility: Physical file size ratios are high-value diagnostics for identifying packed or encrypted malware that code-only analysis might miss.
* Model Robustness: While KNN achieved a 93% clean accuracy, its performance plummeted to 65% under noise; conversely, Random Forest maintained 98.7% accuracy during robustness testing.
* Class Imbalance: Ensemble models with built-in class weight balancing were mandatory for correctly identifying rare families like Class 4, which were consistently misclassified by simpler proximity-based models.

##  References
* Ronen, R., et al. (2018). "Microsoft Malware Classification Challenge." 
* Ahmadi, M., et al. (2016). "Novel Feature Extraction, Selection and Fusion for Effective Malware Family Classification."
* Narayanan, B. N., et al. (2021). "Empirical Study on Microsoft Malware Classification." IJACSA.
