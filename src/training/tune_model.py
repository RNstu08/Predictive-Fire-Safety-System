# src/training/tune_model.py

import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.xgboost
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay,
    make_scorer, # To create custom scorers for RandomizedSearchCV
    recall_score,
    precision_score,
    f1_score,
    accuracy_score
)
from scipy.stats import uniform, randint # For defining parameter distributions
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
PROCESSED_DATA_DIR = os.path.join("data", "processed")
# *** UPDATE FILENAME if your processed file from Phase 3 has a different name ***
PROCESSED_FILENAME = "processed_dummy_fault_RackA2_Module01_200rows.csv"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, PROCESSED_FILENAME)

TARGET_COLUMN = 'Hazard_Label'
EXCLUDE_COLS = ['Timestamp', 'Module_ID', 'Rack_ID', TARGET_COLUMN]
TEST_SIZE = 0.2
RANDOM_STATE = 42
MLFLOW_EXPERIMENT_NAME = "HVS_Fire_Prediction_Tuning" # New experiment for tuning runs

# --- RandomizedSearch Configuration ---
N_ITER_SEARCH = 20 # Number of parameter settings that are sampled. Increase for more thorough search.
CV_FOLDS = 3 # Number of cross-validation folds

# Define the parameter distribution for Randomized Search (Example for XGBoost)
# These ranges should be adjusted based on baseline model performance and domain knowledge
PARAM_DISTRIBUTIONS = {
    'n_estimators': randint(100, 500), # Number of trees
    'learning_rate': uniform(0.01, 0.3), # Step size shrinkage
    'max_depth': randint(3, 10), # Maximum depth of a tree
    'subsample': uniform(0.6, 0.4), # Fraction of samples used per tree (0.6 to 1.0)
    'colsample_bytree': uniform(0.6, 0.4), # Fraction of features used per tree (0.6 to 1.0)
    'gamma': uniform(0, 0.5), # Minimum loss reduction required to make a further partition
    'reg_alpha': uniform(0, 1), # L1 regularization term on weights
    'reg_lambda': uniform(0, 1) # L2 regularization term on weights
}

# Define the primary metric to optimize during search (e.g., F1 score for the positive class)
# We need make_scorer because RandomizedSearchCV needs a score, not just a metric function
SCORING_METRIC = make_scorer(f1_score, pos_label=1) # Optimize for F1 score of the hazardous class

# Define artifact paths
TUNED_MODEL_ARTIFACT_PATH = "tuned_xgboost_model" # MLflow logs models in directories
SCALER_ARTIFACT_PATH = "scaler.joblib"
CONFUSION_MATRIX_PLOT_PATH = "tuned_confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = "tuned_classification_report.txt"

# --- Helper Functions ---
# load_processed_data, prepare_data_for_training, split_data, scale_features are similar to train_mlflow.py
# (Ensure scale_features saves/logs the scaler correctly if needed outside this script)

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
    print(f"  Features ({len(features)}): {features[:5]}...") # Print first 5 features
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

def scale_features(X_train, X_test, log_to_mlflow=False, artifact_path="scaler.joblib"):
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  Scaling complete.")
    if log_to_mlflow:
        joblib.dump(scaler, artifact_path)
        mlflow.log_artifact(artifact_path)
        print(f"  Logged scaler artifact: {artifact_path}")
        try: os.remove(artifact_path)
        except OSError as e: print(f"  Warning: Could not remove local scaler file: {e}")
    return X_train_scaled, X_test_scaled, scaler

def evaluate_model_and_log_tuning(y_true, y_pred, y_pred_proba, best_params, model_name="Tuned Model"):
    """Evaluates the final tuned model and logs results."""
    print(f"\n--- Evaluating Final Tuned Model: {model_name} ---")
    mlflow.set_tag("model_type", "XGBoost_Tuned") # Example tag
    mlflow.log_params(best_params) # Log the best parameters found

    # Calculate metrics
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

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report_str)
    if roc_auc is not None: print(f"AUC-ROC Score: {roc_auc:.4f}")
    if pr_auc is not None: print(f"AUC-PR Score: {pr_auc:.4f}")

    # Log metrics
    print("  Logging final metrics...")
    mlflow.log_metrics({
        "final_accuracy": accuracy, "final_precision_class1": precision_1,
        "final_recall_class1": recall_1, "final_f1_score_class1": f1_1,
        "final_precision_macro": precision_macro, "final_recall_macro": recall_macro,
        "final_f1_score_macro": f1_macro,
        "final_roc_auc": roc_auc if roc_auc is not None else 0, # Log 0 if not calculable
        "final_pr_auc": pr_auc if pr_auc is not None else 0
    })
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0], 0, 0, 0) if cm.shape == (1,1) else (0,0,0,0)
    mlflow.log_metrics({ "final_true_negatives": tn, "final_false_positives": fp,
                         "final_false_negatives": fn, "final_true_positives": tp })

    # Log artifacts
    print("  Logging final artifacts...")
    try: # Confusion Matrix
        fig, ax = plt.subplots(); disp = ConfusionMatrixDisplay(cm); disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}'); plt.savefig(CONFUSION_MATRIX_PLOT_PATH); plt.close(fig)
        mlflow.log_artifact(CONFUSION_MATRIX_PLOT_PATH); os.remove(CONFUSION_MATRIX_PLOT_PATH)
        print(f"    Logged confusion matrix plot: {CONFUSION_MATRIX_PLOT_PATH}")
    except Exception as e: print(f"    Warning: Could not log confusion matrix plot: {e}")
    try: # Classification Report
        with open(CLASSIFICATION_REPORT_PATH, "w") as f: f.write(report_str)
        mlflow.log_artifact(CLASSIFICATION_REPORT_PATH); os.remove(CLASSIFICATION_REPORT_PATH)
        print(f"    Logged classification report: {CLASSIFICATION_REPORT_PATH}")
    except Exception as e: print(f"    Warning: Could not log classification report: {e}")


# --- Main Tuning Function ---

def tune_xgboost_model(data_path):
    """Tunes XGBoost using RandomizedSearchCV and logs with MLflow."""
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

    # Start the main run for this tuning process
    with mlflow.start_run(run_name="XGBoost_RandomSearch_Tuning") as run:
        run_id = run.info.run_id
        print(f"\n--- Starting MLflow Run for XGBoost Tuning (ID: {run_id}) ---")
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_param("n_iter_search", N_ITER_SEARCH)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("scoring_metric", "f1_pos_label") # Describe scorer

        # 4. Scale Features and Log Scaler
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, log_to_mlflow=True, artifact_path=SCALER_ARTIFACT_PATH)

        # 5. Calculate Class Weights (still needed for the base estimator in search)
        neg_count = np.sum(y_train == 0); pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        print(f"  Using scale_pos_weight: {scale_pos_weight:.2f}")

        # 6. Setup Randomized Search
        xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                            scale_pos_weight=scale_pos_weight, use_label_encoder=False,
                            random_state=RANDOM_STATE, n_jobs=-1)

        random_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=N_ITER_SEARCH, # Number of parameter settings that are sampled
            scoring=SCORING_METRIC, # Optimize for F1 score of class 1
            n_jobs=-1, # Use all available cores for search parallelism
            cv=StratifiedShuffleSplit(n_splits=CV_FOLDS, test_size=0.2, random_state=RANDOM_STATE), # Stratified CV
            random_state=RANDOM_STATE,
            verbose=1 # Print progress
        )

        # 7. Run Search
        print(f"\nStarting RandomizedSearchCV (n_iter={N_ITER_SEARCH}, cv={CV_FOLDS})...")
        random_search.fit(X_train_scaled, y_train)
        print("RandomizedSearchCV finished.")

        # 8. Log Best Results from Search
        print(f"\nBest parameters found: {random_search.best_params_}")
        print(f"Best cross-validation score ({SCORING_METRIC}): {random_search.best_score_:.4f}")
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_cv_f1_score_class1", random_search.best_score_)

        # 9. Evaluate Best Model on Test Set
        best_model = random_search.best_estimator_
        y_pred_test = best_model.predict(X_test_scaled)
        y_pred_proba_test = best_model.predict_proba(X_test_scaled)[:, 1]

        evaluate_model_and_log_tuning(y_test, y_pred_test, y_pred_proba_test,
                                      best_params=random_search.best_params_,
                                      model_name="XGBoost_Tuned_Final")

        # 10. Log the final tuned model
        print("  Logging final tuned model...")
        model_info = mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path=TUNED_MODEL_ARTIFACT_PATH,
            input_example=pd.DataFrame(X_train_scaled[:5,:], columns=features)
        )
        print(f"    Logged final tuned model artifact path: {TUNED_MODEL_ARTIFACT_PATH}, URI: {model_info.model_uri}")

        print(f"--- Finished MLflow Run for XGBoost Tuning (ID: {run_id}) ---")


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting XGBoost Hyperparameter Tuning Pipeline with MLflow...")
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_file_path = os.path.join(PROJECT_ROOT, PROCESSED_DATA_PATH)

    try:
        tune_xgboost_model(data_file_path)
        print("\nTuning pipeline finished successfully.")
        print(f"\nMLflow results logged. Run 'mlflow ui' in the terminal from the project root ('{PROJECT_ROOT}') to view the UI.")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the processed data file exists. Run the preprocessing pipeline (Phase 3) first.")
    except ModuleNotFoundError as e:
         print(f"\nError: {e}. Please install required packages (e.g., 'pip install mlflow xgboost scikit-learn pandas joblib matplotlib scipy').") # Added scipy
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

