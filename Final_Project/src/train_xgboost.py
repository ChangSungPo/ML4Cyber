import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

def main():
    # 1. Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'features_with_metadata.csv')

    print("1. Loading upgraded feature matrix (Opcodes + File Sizes)...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=['Id', 'Class'])
    y = df['Class']

    # 2. Train/Test Split
    print("2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train the Model
    print("3. Training XGBoost Classifier...")
    # XGBoost strictly requires class labels to start at 0, so we shift them down by 1
    y_train_xgb = y_train - 1
    y_test_xgb = y_test - 1
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        random_state=42, 
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train_xgb)

    # 4. Evaluate
    print("4. Evaluating model...")
    y_pred_xgb = xgb_model.predict(X_test)

    # Shift labels back up by 1 so the output matches our Kaggle 1-9 classes
    y_test_original = y_test_xgb + 1
    y_pred_original = y_pred_xgb + 1

    # 5. Print Results
    print("\n" + "="*50)
    print("           XGBOOST MODEL RESULTS (WEEK 5)")
    print("="*50)
    print(f"Overall Accuracy: {accuracy_score(y_test_original, y_pred_original) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test_original, y_pred_original))

if __name__ == '__main__':
    main()