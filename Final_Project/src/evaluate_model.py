import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import os

def main():
    # 1. Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'asm_features.csv')
    
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Id', 'Class'])
    y = df['Class']

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 4. Create Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(df['Class'].unique())

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Baseline Model: Confusion Matrix')
    plt.ylabel('Actual Malware Family')
    plt.xlabel('Predicted Malware Family')
    
    # 5. Save
    output_path = os.path.join(project_root, 'reports', 'figures', 'confusion_matrix.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Confusion Matrix saved to: {output_path}")

if __name__ == '__main__':
    main()
