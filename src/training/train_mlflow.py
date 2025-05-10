# src/training/train_mlflow.py
# (Consider renaming the file to reflect MLflow integration)

import pandas as pd
import numpy as np
import os
import joblib # For saving scaler
import mlflow # Import MLflow
import mlflow.sklearn # For logging scikit-learn models
import mlflow.xgboost # For logging XGBoost models
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
    ConfusionMatrixDisplay,
    f1_score, # Add F1 explicitly for logging
    precision_score, # Add Precision explicitly for logging
    recall_score # Add Recall explicitly for logging
)
import matplotlib.pyplot as plt
import warnings

# Suppress specific warnings if needed (e.g., from XGBoost or sklearn)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Configuration ---
PROCESSED_DATA_DIR = os.path.join("data", "processed")
# *** UPDATE FILENAME if your processed file from Phase 3 has a different name ***
# Using the dummy data file name from previous steps
PROCESSED_FILENAME = "processed_dummy_fault_RackA2_Module01_200rows.csv"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, PROCESSED_FILENAME)

TARGET_COLUMN = 'Hazard_Label'
EXCLUDE_COLS = ['Timestamp', 'Module_ID', 'Rack_ID', TARGET_COLUMN]
TEST_SIZE = 0.2
RANDOM_STATE = 42
MLFLOW_EXPERIMENT_NAME = "HVS_Fire_Prediction_Baseline" # Name for the experiment in MLflow UI

# Define artifact paths (within MLflow run)
SCALER_ARTIFACT_PATH = "scaler.joblib"
CONFUSION_MATRIX_PLOT_PATH = "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = "classification_report.txt"

# --- Helper Functions ---
# load_processed_data, prepare_data_for_training, split_data remain the same as training_script_v1

def load_processed_data(file_path):
    """Loads the processed data."""
    print(f"Loading processed data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data file not found at {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded processed data shape: {df.shape}")
    if TARGET_COLUMN not in df.columns:
         raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the data.")
    return df

def prepare_data_for_training(df):
    """Separates features (X) and target (y), identifies feature names."""
    print("Preparing data: Selecting features (X) and target (y)...")
    y = df[TARGET_COLUMN]
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    features = [col for col in numeric_cols if col not in EXCLUDE_COLS]
    X = df[features]
    print(f"  Number of features: {len(features)}")
    print(f"  Target variable: '{TARGET_COLUMN}'")
    if X.isnull().sum().sum() > 0:
        print("Warning: NaNs found in feature set X before splitting. Filling with 0.")
        X = X.fillna(0)
    return X, y, features

def split_data(X, y, test_size=0.2, random_state=42):
    """Performs a stratified train-test split."""
    print(f"Splitting data into train/test sets (Test size: {test_size*100}%)...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print(f"  Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"  Test set shape: X={X_test.shape}, y={y_test.shape}")
    print(f"  Hazard Label distribution in training set:\n{y_train.value_counts(normalize=True)}")
    print(f"  Hazard Label distribution in test set:\n{y_test.value_counts(normalize=True)}")
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, log_to_mlflow=False):
    """Applies StandardScaler and optionally logs the scaler artifact."""
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  Scaling complete.")

    if log_to_mlflow:
        # Save the scaler locally first
        joblib.dump(scaler, SCALER_ARTIFACT_PATH)
        # Log the scaler as an artifact in MLflow
        mlflow.log_artifact(SCALER_ARTIFACT_PATH)
        print(f"  Logged scaler artifact: {SCALER_ARTIFACT_PATH}")
        # Clean up local file after logging
        os.remove(SCALER_ARTIFACT_PATH)

    return X_train_scaled, X_test_scaled, scaler

def evaluate_model_and_log(y_true, y_pred, y_pred_proba, model_name="Model"):
    """Calculates, prints, and logs evaluation metrics and artifacts to MLflow."""
    print(f"\n--- Evaluating & Logging: {model_name} ---")
    mlflow.set_tag("model_name", model_name) # Tag the run with model name

    # --- Calculate Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_str = classification_report(y_true, y_pred)
    # Extract metrics for class 1 (Hazardous) specifically for logging
    precision_1 = report_dict.get('1', {}).get('precision', 0)
    recall_1 = report_dict.get('1', {}).get('recall', 0)
    f1_1 = report_dict.get('1', {}).get('f1-score', 0)
    # Also log macro averages
    precision_macro = report_dict.get('macro avg', {}).get('precision', 0)
    recall_macro = report_dict.get('macro avg', {}).get('recall', 0)
    f1_macro = report_dict.get('macro avg', {}).get('f1-score', 0)

    roc_auc = None
    pr_auc = None
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError as e:
            print(f"Could not calculate ROC AUC: {e}") # Handle cases with only one class in y_true
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)
        except ValueError as e:
             print(f"Could not calculate PR AUC: {e}")

    # --- Print Metrics ---
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("\nClassification Report:")
    print(report_str)
    if roc_auc is not None: print(f"AUC-ROC Score: {roc_auc:.4f}")
    if pr_auc is not None: print(f"AUC-PR Score: {pr_auc:.4f}")

    # --- Log Metrics to MLflow ---
    print("  Logging metrics to MLflow...")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_class1", precision_1)
    mlflow.log_metric("recall_class1", recall_1)
    mlflow.log_metric("f1_score_class1", f1_1)
    mlflow.log_metric("precision_macro", precision_macro)
    mlflow.log_metric("recall_macro", recall_macro)
    mlflow.log_metric("f1_score_macro", f1_macro)
    if roc_auc is not None: mlflow.log_metric("roc_auc", roc_auc)
    if pr_auc is not None: mlflow.log_metric("pr_auc", pr_auc)

    # --- Log Artifacts to MLflow ---
    print("  Logging artifacts to MLflow...")
    # 1. Confusion Matrix Plot
    try:
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}')
        plt.savefig(CONFUSION_MATRIX_PLOT_PATH) # Save plot locally
        plt.close(fig) # Close plot to free memory
        mlflow.log_artifact(CONFUSION_MATRIX_PLOT_PATH) # Log the plot file
        print(f"    Logged confusion matrix plot: {CONFUSION_MATRIX_PLOT_PATH}")
        os.remove(CONFUSION_MATRIX_PLOT_PATH) # Clean up local file
    except Exception as e:
        print(f"    Could not save/log confusion matrix plot: {e}")

    # 2. Classification Report Text File
    try:
        with open(CLASSIFICATION_REPORT_PATH, "w") as f:
            f.write(f"Classification Report for {model_name}:\n\n")
            f.write(report_str)
        mlflow.log_artifact(CLASSIFICATION_REPORT_PATH)
        print(f"    Logged classification report: {CLASSIFICATION_REPORT_PATH}")
        os.remove(CLASSIFICATION_REPORT_PATH) # Clean up local file
    except Exception as e:
        print(f"    Could not save/log classification report: {e}")

    # Return key metrics if needed
    return accuracy, recall_1, precision_1, f1_1, roc_auc, pr_auc


# --- Main Training Function ---

def train_and_evaluate_models_with_mlflow(data_path):
    """Loads data, preprocesses, trains, evaluates, and logs models with MLflow."""
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

    # Start a parent run for the overall process (optional, but good practice)
    with mlflow.start_run(run_name="Preprocessing_and_Scaling") as parent_run:
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("num_features_initial", len(features))
        mlflow.log_param("training_rows", X_train.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])

        # 4. Scale Features and Log Scaler
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, log_to_mlflow=True)
        mlflow.set_tag("pipeline_step", "scaling_complete")

    # 5. Calculate Class Weights
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = 1
    if pos_count > 0:
        scale_pos_weight = neg_count / pos_count
    print(f"\nCalculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

    # Define Models (parameters can be logged within the loop)
    model_configs = {
        "Logistic Regression": {
            "model": LogisticRegression(solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000),
            "params": {"solver": "liblinear", "class_weight": "balanced", "max_iter": 1000}
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
            "params": {"n_estimators": 100, "class_weight": "balanced"}
        },
        "XGBoost": {
            "model": XGBClassifier(objective='binary:logistic', eval_metric='logloss', scale_pos_weight=scale_pos_weight, use_label_encoder=False, random_state=RANDOM_STATE, n_jobs=-1),
            "params": {"objective": "binary:logistic", "eval_metric": "logloss", "scale_pos_weight": scale_pos_weight}
        }
    }

    results = {}
    trained_models = {}

    # 6. Train and Evaluate Each Model within its own MLflow run
    for name, config in model_configs.items():
        print(f"\n--- Starting MLflow Run for: {name} ---")
        # Start a nested run for each model under the main experiment
        with mlflow.start_run(run_name=name, nested=True) as run:
            model = config["model"]
            params = config["params"]
            run_id = run.info.run_id
            print(f"  MLflow Run ID: {run_id}")

            # Log parameters
            print("  Logging parameters...")
            mlflow.log_params(params)
            mlflow.log_param("feature_count", len(features)) # Log number of features used
            mlflow.log_param("scaler", "StandardScaler")

            # Train model
            print(f"  Training {name}...")
            model.fit(X_train_scaled, y_train)
            print("  Training complete.")

            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = None
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Evaluate and Log Metrics/Artifacts using helper function
            accuracy, recall_1, precision_1, f1_1, roc_auc, pr_auc = evaluate_model_and_log(
                y_test, y_pred, y_pred_proba, model_name=name
            )
            results[name] = {'Accuracy': accuracy, 'Recall_1': recall_1, 'Precision_1': precision_1, 'F1_1': f1_1, 'ROC_AUC': roc_auc, 'PR_AUC': pr_auc}
            trained_models[name] = model

            # Log the Model using MLflow's model flavors
            print("  Logging model...")
            model_info = None
            if name == "XGBoost":
                # Use log_model for XGBoost specific format
                model_info = mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path=f"{name}_model", # Subdirectory within artifacts
                    # input_example=X_train_scaled[:5,:], # Optional: Log an input example
                    # signature=signature # Optional: Define model signature
                )
            elif name in ["Logistic Regression", "Random Forest"]:
                 # Use log_model for scikit-learn models
                 model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"{name}_model",
                    # input_example=X_train_scaled[:5,:],
                    # signature=signature
                 )
            else:
                 print(f"    Warning: Model flavor for {name} not explicitly handled for logging.")

            if model_info:
                 print(f"    Logged model artifact path: {name}_model, URI: {model_info.model_uri}")

            print(f"--- Finished MLflow Run for: {name} (ID: {run_id}) ---")


    # 7. Compare Results
    print("\n--- Overall Model Comparison ---")
    # Include Recall/Precision/F1 for class 1 in the summary
    results_df = pd.DataFrame({
        'Accuracy': {name: res['Accuracy'] for name, res in results.items()},
        'Recall_Class1': {name: res['Recall_1'] for name, res in results.items()},
        'Precision_Class1': {name: res['Precision_1'] for name, res in results.items()},
        'F1_Class1': {name: res['F1_1'] for name, res in results.items()},
        'ROC_AUC': {name: res['ROC_AUC'] for name, res in results.items()},
        'PR_AUC': {name: res['PR_AUC'] for name, res in results.items()}
    }).sort_values(by='PR_AUC', ascending=False) # Sort by PR_AUC as it's good for imbalance
    print(results_df)

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Model Training and Evaluation Pipeline with MLflow...")
    # Define path relative to project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_file_path = os.path.join(PROJECT_ROOT, PROCESSED_DATA_PATH)

    try:
        train_and_evaluate_models_with_mlflow(data_file_path)
        print("\nPipeline finished successfully.")
        print(f"MLflow results logged. Run 'mlflow ui' in the terminal from '{PROJECT_ROOT}' to view.")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the processed data file exists. Run the preprocessing pipeline first.")
    except ModuleNotFoundError as e:
         print(f"\nError: {e}. Please install required packages (e.g., 'pip install mlflow xgboost scikit-learn pandas joblib matplotlib').")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

