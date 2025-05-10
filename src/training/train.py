# src/training/train.py

import pandas as pd
import numpy as np
import os
import joblib # For saving scaler later if needed
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# --- Configuration ---
PROCESSED_DATA_DIR = os.path.join("data", "processed")
# *** UPDATE FILENAME if your processed file from Phase 3 has a different name ***
PROCESSED_FILENAME = "processed_dummy_fault_RackA2_Module01_200rows.csv"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, PROCESSED_FILENAME)

TARGET_COLUMN = 'Hazard_Label'
# Columns to exclude from features (identifiers, non-numeric, or target)
EXCLUDE_COLS = ['Timestamp', 'Module_ID', 'Rack_ID', TARGET_COLUMN]

TEST_SIZE = 0.2 # 20% of data for testing
RANDOM_STATE = 42 # For reproducibility

# --- Helper Functions ---

def load_processed_data(file_path):
    """Loads the processed data."""
    print(f"Loading processed data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data file not found at {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded processed data shape: {df.shape}")
    # Basic validation
    if TARGET_COLUMN not in df.columns:
         raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the data.")
    return df

def prepare_data_for_training(df):
    """Separates features (X) and target (y), identifies feature names."""
    print("Preparing data: Selecting features (X) and target (y)...")
    y = df[TARGET_COLUMN]
    # Select only numeric columns and exclude identifier/target columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    features = [col for col in numeric_cols if col not in EXCLUDE_COLS]
    X = df[features]
    print(f"  Number of features: {len(features)}")
    print(f"  Target variable: '{TARGET_COLUMN}'")
    # Check for NaNs introduced unexpectedly
    if X.isnull().sum().sum() > 0:
        print("Warning: NaNs found in feature set X before splitting. Consider revisiting preprocessing.")
        # Simple fill for safety, though ideally preprocessing handles this
        X = X.fillna(0)
    return X, y, features

def split_data(X, y, test_size=0.2, random_state=42):
    """Performs a stratified train-test split."""
    print(f"Splitting data into train/test sets (Test size: {test_size*100}%)...")
    # Using StratifiedShuffleSplit to ensure proportions are maintained, especially for imbalance
    # n_splits=1 means we just want one split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Get the indices for the split
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"  Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"  Test set shape: X={X_test.shape}, y={y_test.shape}")
    print(f"  Hazard Label distribution in training set:\n{y_train.value_counts(normalize=True)}")
    print(f"  Hazard Label distribution in test set:\n{y_test.value_counts(normalize=True)}")
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Applies StandardScaler to the feature sets."""
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  Scaling complete.")
    # Note: Consider saving the scaler object (e.g., using joblib) if you need to apply it later for inference
    # joblib.dump(scaler, 'scaler.joblib')
    return X_train_scaled, X_test_scaled, scaler

def evaluate_model(y_true, y_pred, y_pred_proba, model_name="Model"):
    """Calculates and prints evaluation metrics."""
    print(f"\n--- Evaluating: {model_name} ---")

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")


    # Classification Report
    print("\nClassification Report:")
    # target_names=['Normal (0)', 'Hazardous (1)'] # Optional: for better readability
    report = classification_report(y_true, y_pred) #, target_names=target_names)
    print(report)

    # AUC-ROC
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"AUC-ROC Score: {roc_auc:.4f}")
    else:
        print("AUC-ROC Score: Not available (no probability predictions).")
        roc_auc = None

    # AUC-PR (Precision-Recall Curve) - More informative for imbalanced data
    if y_pred_proba is not None:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        print(f"AUC-PR Score: {pr_auc:.4f}")
    else:
        print("AUC-PR Score: Not available (no probability predictions).")
        pr_auc = None

    return accuracy, report, roc_auc, pr_auc


# --- Main Training Function ---

def train_and_evaluate_models(data_path):
    """Loads data, preprocesses (scaling), trains, and evaluates models."""
    # 1. Load Data
    df = load_processed_data(data_path)
    if df is None: return

    # 2. Prepare Data (X, y separation)
    X, y, features = prepare_data_for_training(df)

    # 3. Split Data (Stratified)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 4. Scale Features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    # Note: For models like RF and XGBoost, scaling isn't strictly necessary,
    # but it's crucial for Logistic Regression. We scale for all for consistency here.

    # 5. Handle Class Imbalance - Calculate Weights
    # Calculate scale_pos_weight for XGBoost (ratio of negatives to positives)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    if pos_count == 0:
        print("Warning: No positive samples found in the training data. Cannot calculate scale_pos_weight.")
        scale_pos_weight = 1 # Default if no positives
    else:
        scale_pos_weight = neg_count / pos_count
    print(f"\nCalculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

    # Define Models
    models = {
        "Logistic Regression": LogisticRegression(
            solver='liblinear', # Good for smaller datasets
            class_weight='balanced', # Handles imbalance
            random_state=RANDOM_STATE,
            max_iter=1000 # Increase max_iter for convergence
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, # Number of trees
            class_weight='balanced', # Handles imbalance
            random_state=RANDOM_STATE,
            n_jobs=-1 # Use all available CPU cores
        ),
        "XGBoost": XGBClassifier(
            objective='binary:logistic', # Binary classification
            eval_metric='logloss', # Evaluation metric for training
            scale_pos_weight=scale_pos_weight, # Handles imbalance
            use_label_encoder=False, # Recommended setting
            random_state=RANDOM_STATE,
            n_jobs=-1 # Use all available CPU cores
        )
    }

    results = {}

    # 6. Train and Evaluate Each Model
    for name, model in models.items():
        print(f"\n--- Training: {name} ---")
        # Train on scaled data
        model.fit(X_train_scaled, y_train)
        print("  Training complete.")

        # Make predictions on the scaled test data
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            # Get probabilities for the positive class (label 1)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate
        accuracy, report, roc_auc, pr_auc = evaluate_model(y_test, y_pred, y_pred_proba, model_name=name)
        results[name] = {'Accuracy': accuracy, 'ROC_AUC': roc_auc, 'PR_AUC': pr_auc, 'Classification Report': report}

    # 7. Compare Results (Optional: Print summary table)
    print("\n--- Overall Model Comparison ---")
    results_df = pd.DataFrame({
        'Accuracy': {name: res['Accuracy'] for name, res in results.items()},
        'ROC_AUC': {name: res['ROC_AUC'] for name, res in results.items()},
        'PR_AUC': {name: res['PR_AUC'] for name, res in results.items()}
    })
    print(results_df)
    print("\nNote: Focus on ROC_AUC, PR_AUC, and especially Recall/Precision/F1 for class 1 in the Classification Reports above for imbalanced data.")

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Model Training and Evaluation Pipeline...")
    # Define path relative to project root (assuming this script is in src/training)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_file_path = os.path.join(PROJECT_ROOT, PROCESSED_DATA_PATH)

    try:
        train_and_evaluate_models(data_file_path)
        print("\nPipeline finished successfully.")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the processed data file exists. Run the preprocessing pipeline first.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

