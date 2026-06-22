import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def get_f1_scores(data_path, drop_cols):
    df = pd.read_csv(data_path)
    X = df.drop(columns=drop_cols)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return f1_score(y_test, y_pred, average=None), sorted(df['Class'].unique())

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 1. Get Champion Scores (Opcodes + File Sizes)
    print("Training Champion Model (Metadata)...")
    metadata_path = os.path.join(project_root, 'data', 'features_with_metadata.csv')
    meta_f1, classes = get_f1_scores(metadata_path, ['Id', 'Class'])
    
    # 2. Get Final Scores (Added N-Grams)
    print("Training Final Model (N-Grams)...")
    final_path = os.path.join(project_root, 'data', 'features_final.csv')
    final_f1, _ = get_f1_scores(final_path, ['Id', 'Class'])
    
    # 3. Plotting the Comparison
    print("Generating Comparison Plot...")
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Champion in Green, Final (with noise) in Red
    rects1 = ax.bar(x - width/2, meta_f1, width, label='Best model (Opcodes + Metadata)', color='#2CA02C')
    rects2 = ax.bar(x + width/2, final_f1, width, label='Final (Added N-Grams)', color='#D62728')

    ax.set_ylabel('F1-Score (Higher is Better)')
    ax.set_title('Bi-gram Feature Impact on Class-wise F1 Scores')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1])
    
    # Add a grid for easier reading
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    output_path = os.path.join(project_root, 'reports', 'figures', 'ngram_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\nSuccess! Open this file to see the dropoff: {output_path}")

if __name__ == '__main__':
    main()
