import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'features_final.csv')

    print("Loading FINAL feature matrix (Opcodes + Metadata + N-Grams)...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=['Id', 'Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    print("\n" + "="*50)
    print("           FINAL MODEL RESULTS (ALL FEATURES)")
    print("="*50)
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()