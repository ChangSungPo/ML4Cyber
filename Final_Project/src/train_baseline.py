import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    # 1. Bulletproof Pathing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'asm_features.csv')

    print("1. Loading feature matrix...")
    df = pd.read_csv(data_path)

    # 2. Separate Features (X) and Labels (y)
    X = df.drop(columns=['Id', 'Class'])
    y = df['Class']

    # 3. Train/Test Split (80/20)
    print("2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Initialize and Train the Model
    print("3. Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    # 5. Make Predictions & Evaluate
    print("4. Evaluating model...")
    y_pred = rf_model.predict(X_test)

    # 6. Print Results
    print("\n" + "="*50)
    print("             BASELINE MODEL RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report (Focus on F1-Score):")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
