# src/api/services/prediction_service.py

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os
from typing import List, Dict, Tuple

# --- Configuration ---
BEST_MODEL_RUN_ID = os.environ.get("BEST_MLFLOW_RUN_ID", "RUN_ID_NOT_SET")
# --- IMPORTANT: Set your Experiment ID here ---
# This is the ID of the experiment under which the BEST_MODEL_RUN_ID was logged.
# You found this in your error message: /mlruns/EXPERIMENT_ID/RUN_ID/...
# For run db41593df050433ab1d871bf63584505, your Experiment ID was 280465920837496359
MLFLOW_EXPERIMENT_ID = "280465920837496359" # <<< UPDATE THIS IF DIFFERENT FOR YOUR CHOSEN RUN

MODEL = None
SCALER = None
MODEL_VERSION = None
FEATURE_NAMES = None

SCALER_ARTIFACT_FILENAME = "scaler.joblib"
# --- IMPORTANT: Verify this from your MLflow UI for the chosen BEST_MODEL_RUN_ID ---
# This is the name of the directory where your model was logged within the run's artifacts.
# Examples: "XGBoost_Tuned_Final", "xgboost_smote_model", "Random Forest_model"
MODEL_ARTIFACT_SUBPATH = "tuned_xgboost_model" # <<< VERIFY AND UPDATE THIS

def load_model_and_scaler():
    """Loads the scaler and model artifacts directly from the mounted mlruns volume."""
    global MODEL, SCALER, MODEL_VERSION, FEATURE_NAMES
    
    if MODEL is not None and SCALER is not None:
        print("Model and scaler already loaded.")
        return True

    print("Loading model and scaler using direct path construction...")
    if BEST_MODEL_RUN_ID == "RUN_ID_NOT_SET" or BEST_MODEL_RUN_ID == "":
        print("ERROR: BEST_MLFLOW_RUN_ID environment variable is not set or is empty.")
        return False
    if MLFLOW_EXPERIMENT_ID == "" or MLFLOW_EXPERIMENT_ID is None:
        print("ERROR: MLFLOW_EXPERIMENT_ID is not set in the script.")
        return False

    # Path to the root of the mounted mlruns directory inside the container
    # This is set by ENV MLFLOW_TRACKING_URI="file:///app/mlruns" in Dockerfile,
    # and the volume mount -v ${PWD}/mlruns:/app/mlruns
    base_mlruns_path_in_container = "/app/mlruns"

    run_artifacts_path = os.path.join(
        base_mlruns_path_in_container,
        MLFLOW_EXPERIMENT_ID,
        BEST_MODEL_RUN_ID,
        "artifacts"
    )
    print(f"  Constructed base artifacts path in container: {run_artifacts_path}")

    try:
        # 1. Load Scaler
        scaler_full_path_in_container = os.path.join(run_artifacts_path, SCALER_ARTIFACT_FILENAME)
        print(f"  Attempting to load scaler from: {scaler_full_path_in_container}")
        if not os.path.exists(scaler_full_path_in_container):
            raise FileNotFoundError(f"Scaler file not found at {scaler_full_path_in_container}")
        SCALER = joblib.load(scaler_full_path_in_container)
        print("  Scaler loaded successfully.")

        # 2. Load Model
        model_dir_full_path_in_container = os.path.join(run_artifacts_path, MODEL_ARTIFACT_SUBPATH)
        print(f"  Attempting to load model from directory: {model_dir_full_path_in_container}")
        if not os.path.isdir(model_dir_full_path_in_container):
            raise FileNotFoundError(f"Model directory not found at {model_dir_full_path_in_container}")
        
        # Determine model flavor based on the artifact subpath or a known convention
        if "xgboost" in MODEL_ARTIFACT_SUBPATH.lower() or "xgb" in MODEL_ARTIFACT_SUBPATH.lower():
            MODEL = mlflow.xgboost.load_model(model_dir_full_path_in_container)
        elif "sklearn" in MODEL_ARTIFACT_SUBPATH.lower() or \
             "logistic" in MODEL_ARTIFACT_SUBPATH.lower() or \
             "forest" in MODEL_ARTIFACT_SUBPATH.lower():
            MODEL = mlflow.sklearn.load_model(model_dir_full_path_in_container)
        else:
            # Attempt a generic load if flavor is unclear, or raise error
            try:
                MODEL = mlflow.pyfunc.load_model(model_dir_full_path_in_container)
                print("  Model loaded using generic mlflow.pyfunc.load_model.")
            except Exception as pyfunc_e:
                raise ValueError(f"Unknown model type for artifact path: {MODEL_ARTIFACT_SUBPATH}. Cannot determine MLflow flavor. Pyfunc error: {pyfunc_e}")
        print("  Model loaded successfully.")

        MODEL_VERSION = BEST_MODEL_RUN_ID

        if hasattr(SCALER, 'feature_names_in_'):
            FEATURE_NAMES = SCALER.feature_names_in_
        elif hasattr(MODEL, 'feature_names_in_'): # For some model types
             FEATURE_NAMES = MODEL.feature_names_in_
        else: # Fallback if feature names are not directly available
             print("Warning: Could not automatically determine feature names from scaler or model. Ensure input data has correct features in the correct order.")
             # You might need to load them from another artifact or define them if this happens.
             # For now, preprocess_input will try to use all numeric columns if FEATURE_NAMES is None.
             FEATURE_NAMES = None
        
        if FEATURE_NAMES is not None:
            print(f"  Loaded {len(FEATURE_NAMES)} feature names.")

        print("Model and scaler loaded successfully using direct paths from mounted volume.")
        return True

    except Exception as e:
        print(f"Error loading model/scaler using direct path strategy for run {BEST_MODEL_RUN_ID}: {e}")
        import traceback
        traceback.print_exc()
        MODEL = None; SCALER = None; MODEL_VERSION = None; FEATURE_NAMES = None
        return False

# preprocess_input and predict_hazard functions remain the same
def preprocess_input(data: List[Dict]) -> Tuple[pd.DataFrame, List[str | None]]:
    print(f"Preprocessing {len(data)} input readings...")
    if not data: return pd.DataFrame(), []
    df = pd.DataFrame(data)
    module_ids = df.get('Module_ID', pd.Series([None] * len(df))).tolist()

    if FEATURE_NAMES is not None:
        input_cols = df.columns.tolist()
        missing_model_features = [col for col in FEATURE_NAMES if col not in input_cols]
        if missing_model_features:
            raise ValueError(f"Input data is missing required features: {missing_model_features}")
        try:
            df_features_ordered = df[FEATURE_NAMES]
        except KeyError as e:
            raise ValueError(f"Error selecting features for preprocessing. Expected features not found: {e}")
    else:
        print("Warning: FEATURE_NAMES not available. Using all numeric columns from input, excluding known IDs. Order might be incorrect.")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Basic exclusion, might need to be more robust if other non-feature numerics exist
        df_features_ordered = df[[col for col in numeric_cols if col not in ['Module_ID', 'Rack_ID', 'Timestamp']]]
        if df_features_ordered.empty:
            raise ValueError("No numeric features found for preprocessing after basic exclusion.")

    if df_features_ordered.isnull().sum().sum() > 0:
        print("  Input data contains NaNs. Filling with 0.")
        df_features_ordered = df_features_ordered.fillna(0)
    if SCALER is None: raise RuntimeError("Scaler is not loaded.")
    
    print(f"  Scaling {df_features_ordered.shape[1]} input features...")
    try:
        scaled_features = SCALER.transform(df_features_ordered)
    except ValueError as e:
        scaler_expected_features = SCALER.n_features_in_ if hasattr(SCALER, 'n_features_in_') else 'N/A'
        scaler_actual_names = SCALER.feature_names_in_ if hasattr(SCALER, 'feature_names_in_') else 'N/A'
        raise ValueError(f"Error during scaling. Input has {df_features_ordered.shape[1]} features, scaler expects {scaler_expected_features}. Scaler features: {scaler_actual_names}. Input features: {df_features_ordered.columns.tolist()}. Error: {e}")
    print("  Preprocessing complete.")
    return scaled_features, module_ids

def predict_hazard(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    if MODEL is None: raise RuntimeError("Model is not loaded.")
    print(f"Making predictions on {features.shape[0]} samples with {features.shape[1]} features...")
    predictions = MODEL.predict(features)
    probabilities = None
    if hasattr(MODEL, "predict_proba"):
        probabilities = MODEL.predict_proba(features)[:, 1]
        print("  Generated prediction probabilities.")
    else: print("  Model does not support probability prediction.")
    print("  Prediction complete.")
    return predictions, probabilities

load_model_and_scaler()
