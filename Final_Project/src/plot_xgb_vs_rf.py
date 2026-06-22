import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'features_with_metadata.csv')

    print("Loading data...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Id', 'Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Train Random Forest (The Reigning Champ)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred, average=None)

    # 2. Train XGBoost (The Challenger)
    print("Training XGBoost...")
    y_train_xgb = y_train - 1
    y_test_xgb = y_test - 1
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train_xgb)
    xgb_pred = xgb_model.predict(X_test)
    xgb_f1 = f1_score(y_test_xgb, xgb_pred, average=None)

    # 3. Plotting the Showdown
    print("Generating Comparison Plot...")
    classes = sorted(df['Class'].unique())
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # RF in Green (Winner), XGBoost in Red/Orange (Loser)
    rects1 = ax.bar(x - width/2, rf_f1, width, label='Random Forest (Balanced)', color='#2CA02C')
    rects2 = ax.bar(x + width/2, xgb_f1, width, label='XGBoost (Default)', color='#D62728')

    ax.set_ylabel('F1-Score (Higher is Better)')
    ax.set_title('rf_vs_xgb_comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Save
    output_path = os.path.join(project_root, 'reports', 'figures', 'rf_vs_xgb_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")

if __name__ == '__main__':
    main()