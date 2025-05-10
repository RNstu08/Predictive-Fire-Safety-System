# src/training/train_smote.py

import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.xgboost
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
# --- Import SMOTE ---
from imblearn.over_sampling import SMOTE
# --- Import pipeline for combining SMOTE and model (optional but good practice) ---
# from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
PROCESSED_DATA_DIR = os.path.join("data", "processed")
# *** UPDATE FILENAME if your processed file from Phase 3 has a different name ***
PROCESSED_FILENAME = "processed_dummy_fault_RackA2_Module01_200rows.csv" # Make sure this exists!
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, PROCESSED_FILENAME)

TARGET_COLUMN = 'Hazard_Label'
EXCLUDE_COLS = ['Timestamp', 'Module_ID', 'Rack_ID', TARGET_COLUMN]
TEST_SIZE = 0.2
RANDOM_STATE = 42
MLFLOW_EXPERIMENT_NAME = "HVS_Fire_Prediction_SMOTE" # New experiment for SMOTE runs

# SMOTE configuration
SMOTE_K_NEIGHBORS = 5 # Default value for SMOTE

# Define artifact paths
SCALER_ARTIFACT_PATH = "scaler_smote_run.joblib" # Use different names if needed
CONFUSION_MATRIX_PLOT_PATH = "smote_confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = "smote_classification_report.txt"
MODEL_ARTIFACT_PATH = "xgboost_smote_model"

# --- Helper Functions ---
# load_processed_data, prepare_data_for_training, split_data are same as before
def load_processed_data(file_path):
    print(f"Loading processed data from: {file_path}")
    if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded data shape: {df.shape}")
    if TARGET_COLUMN not in df.columns: raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")
    return df

def prepare_data_for_training(df):
    print("Preparing data...")
    y = df[TARGET_COLUMN]
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    features = [col for col in numeric_cols if col not in EXCLUDE_COLS]
    X = df[features]
    print(f"  Features ({len(features)}): {features[:5]}...")
    print(f"  Target: '{TARGET_COLUMN}'")
    if X.isnull().sum().sum() > 0:
        print("Warning: NaNs found in features. Filling with 0."); X = X.fillna(0)
    return X, y, features

def split_data(X, y, test_size=0.2, random_state=42):
    print(f"Splitting data (Test size: {test_size*100}%)...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print(f"  Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"  Test shape: X={X_test.shape}, y={y_test.shape}")
    return X_train, X_test, y_train, y_test

# scale_features and evaluate_model_and_log are similar, just adjust artifact names if needed
def scale_features(X_train, X_test, log_to_mlflow=False, artifact_path="scaler.joblib"):
    print("Scaling features...")
    scaler = StandardScaler()
    # Fit scaler ONLY on original training data
    scaler.fit(X_train)
    # Transform both original train and test data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  Scaling complete.")
    if log_to_mlflow:
        joblib.dump(scaler, artifact_path)
        mlflow.log_artifact(artifact_path)
        print(f"  Logged scaler artifact: {artifact_path}")
        try: os.remove(artifact_path)
        except OSError as e: print(f"  Warning: Could not remove local scaler file: {e}")
    return X_train_scaled, X_test_scaled, scaler

def evaluate_model_and_log(y_true, y_pred, y_pred_proba, model_name="Model"):
    """Calculates, prints, and logs evaluation metrics and artifacts to MLflow."""
    print(f"\n--- Evaluating & Logging: {model_name} ---")
    mlflow.set_tag("model_name", model_name)

    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, zero_division=0)
    precision_1 = report_dict.get('1', {}).get('precision', 0.0)
    recall_1 = report_dict.get('1', {}).get('recall', 0.0)
    f1_1 = report_dict.get('1', {}).get('f1-score', 0.0)
    precision_macro = report_dict.get('macro avg', {}).get('precision', 0.0)
    recall_macro = report_dict.get('macro avg', {}).get('recall', 0.0)
    f1_macro = report_dict.get('macro avg', {}).get('f1-score', 0.0)
    roc_auc, pr_auc = None, None
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        try: roc_auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError: pass
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)
        except ValueError: pass

    print(f"Accuracy: {accuracy:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report_str)
    if roc_auc is not None: print(f"AUC-ROC Score: {roc_auc:.4f}")
    if pr_auc is not None: print(f"AUC-PR Score: {pr_auc:.4f}")

    print("  Logging metrics to MLflow...")
    mlflow.log_metrics({
        "accuracy": accuracy, "precision_class1": precision_1, "recall_class1": recall_1,
        "f1_score_class1": f1_1, "precision_macro": precision_macro, "recall_macro": recall_macro,
        "f1_score_macro": f1_macro,
        "roc_auc": roc_auc if roc_auc is not None else 0,
        "pr_auc": pr_auc if pr_auc is not None else 0
    })
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0], 0, 0, 0) if cm.shape == (1,1) else (0,0,0,0)
    mlflow.log_metrics({ "true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp })

    print("  Logging artifacts to MLflow...")
    try: # Confusion Matrix
        fig, ax = plt.subplots(); disp = ConfusionMatrixDisplay(cm); disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}'); plt.savefig(CONFUSION_MATRIX_PLOT_PATH); plt.close(fig)
        mlflow.log_artifact(CONFUSION_MATRIX_PLOT_PATH); os.remove(CONFUSION_MATRIX_PLOT_PATH)
        print(f"    Logged confusion matrix plot: {CONFUSION_MATRIX_PLOT_PATH}")
    except Exception as e: print(f"    Warning: Could not log confusion matrix plot: {e}")
    try: # Classification Report
        with open(CLASSIFICATION_REPORT_PATH, "w") as f: f.write(f"Classification Report for {model_name}:\n{report_str}")
        mlflow.log_artifact(CLASSIFICATION_REPORT_PATH); os.remove(CLASSIFICATION_REPORT_PATH)
        print(f"    Logged classification report: {CLASSIFICATION_REPORT_PATH}")
    except Exception as e: print(f"    Warning: Could not log classification report: {e}")

    return accuracy, recall_1, precision_1, f1_1, roc_auc, pr_auc


# --- Main Training Function using SMOTE ---

def train_evaluate_with_smote(data_path):
    """Loads data, preprocesses, applies SMOTE, trains XGBoost, evaluates, and logs."""
    # 1. Load Data
    df = load_processed_data(data_path)
    if df is None: return

    # 2. Prepare Data
    X, y, features = prepare_data_for_training(df)

    # 3. Split Data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # --- MLflow Experiment Setup ---
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"\nUsing MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'")

    # Start MLflow run
    with mlflow.start_run(run_name="XGBoost_with_SMOTE") as run:
        run_id = run.info.run_id
        print(f"\n--- Starting MLflow Run for XGBoost with SMOTE (ID: {run_id}) ---")
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("imbalance_technique", "SMOTE")
        mlflow.log_param("smote_k_neighbors", SMOTE_K_NEIGHBORS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("feature_count", len(features))

        # 4. Scale Features (Fit on original train, transform train & test)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, log_to_mlflow=True, artifact_path=SCALER_ARTIFACT_PATH)

        # 5. Apply SMOTE ONLY to the training data
        print("\nApplying SMOTE to the training data...")
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)
        try:
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
            print(f"  Original training shape: X={X_train_scaled.shape}, y={y_train.shape}")
            print(f"  Shape after SMOTE: X={X_train_smote.shape}, y={y_train_smote.shape}")
            print(f"  Label distribution after SMOTE:\n{pd.Series(y_train_smote).value_counts(normalize=True)}")
            mlflow.log_param("training_rows_before_smote", X_train_scaled.shape[0])
            mlflow.log_param("training_rows_after_smote", X_train_smote.shape[0])
        except Exception as e:
            print(f"Error during SMOTE: {e}. Aborting run.")
            mlflow.set_tag("status", "FAILED_SMOTE")
            return

        # 6. Train XGBoost model (NO scale_pos_weight needed now)
        print("\nTraining XGBoost model on SMOTE'd data...")
        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1
            # Add other hyperparameters here if you tuned them previously,
            # otherwise uses XGBoost defaults
        )
        # Log XGBoost parameters being used
        mlflow.log_params(model.get_params())

        model.fit(X_train_smote, y_train_smote)
        print("  Training complete.")

        # 7. Evaluate on the original (scaled) TEST set
        print("\nEvaluating model on the original TEST set...")
        y_pred_test = model.predict(X_test_scaled)
        y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate and log metrics/artifacts
        evaluate_model_and_log(y_test, y_pred_test, y_pred_proba_test, model_name="XGBoost_with_SMOTE")

        # 8. Log the trained model
        print("  Logging model trained with SMOTE...")
        try:
            model_info = mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path=MODEL_ARTIFACT_PATH,
                input_example=pd.DataFrame(X_train_smote[:5,:], columns=features) # Example from SMOTE'd data
            )
            print(f"    Logged model artifact path: {MODEL_ARTIFACT_PATH}, URI: {model_info.model_uri}")
        except Exception as e:
            print(f"    Error logging SMOTE model artifact: {e}")

        print(f"--- Finished MLflow Run for XGBoost with SMOTE (ID: {run_id}) ---")


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting SMOTE + XGBoost Training Pipeline with MLflow...")
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_file_path = os.path.join(PROJECT_ROOT, PROCESSED_DATA_PATH)

    try:
        train_evaluate_with_smote(data_file_path)
        print("\nPipeline finished successfully.")
        print(f"\nMLflow results logged. Run 'mlflow ui' in the terminal from the project root ('{PROJECT_ROOT}') to view the UI.")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the processed data file exists. Run the preprocessing pipeline (Phase 3) first.")
    except ModuleNotFoundError as e:
         print(f"\nError: {e}. Please install required packages (e.g., 'pip install mlflow xgboost scikit-learn pandas joblib matplotlib imbalanced-learn').") # Added imbalanced-learn
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
