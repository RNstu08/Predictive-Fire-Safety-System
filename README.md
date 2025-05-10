# HVS Predictive Fire Safety System Project
Version: 1.0 (Conceptual Implementation)

## 1. Project Overview
This project aims to develop an end-to-end machine learning system for the early prediction of fire hazards (specifically thermal runaway precursors) in High-Voltage Battery Storage (HVS) systems. It simulates a warehouse environment containing multiple battery modules and includes data generation, exploratory data analysis, preprocessing, feature engineering, model training, evaluation, MLOps practices (experiment tracking with MLflow), API development (Flask), containerization (Docker), deployment concepts (Kubernetes), monitoring concepts, CI/CD automation concepts, and drift/retraining strategies.

**Key Objectives:**
* Predict hazardous conditions (leading to thermal runaway) with high recall and reasonable precision.
* Demonstrate an end-to-end MLOps workflow from data generation to deployment concepts.
* Incorporate best practices like modular code structure, input validation, automated testing concepts, experiment tracking, and containerization.

## 2. Project Structure
The project follows a standard structure for scalable ML projects:
```text
hvs_fire_prediction/
├── .git/                 # Git repository data
├── .github/              # GitHub specific files (e.g., Actions workflows)
│   └── workflows/
│       └── cicd.yaml     # Conceptual CI/CD pipeline definition
├── .gitignore            # Files/directories ignored by Git
├── README.md             # This file
├── requirements.txt      # Python package dependencies
├── notebooks/            # Jupyter notebooks for exploration and analysis
│   └── 01_EDA_Scenario_Analysis.ipynb
├── data/                 # Data files (ignored by Git by default)
│   ├── raw/              # Original generated data (dummy or simulated)
│   ├── processed/        # Cleaned and feature-engineered data
│   └── synthetic_configs/ # (Optional) Configs for data simulation
├── src/                  # Source code for the application
│   ├── data_generation/  # Scripts for generating data
│   │   ├── __init__.py
│   │   ├── simulator.py  # Original physics-based simulator (debugged)
│   │   └── generate_dummy_data.py # Script to generate dummy data (used)
│   ├── preprocessing/    # Data cleaning and feature engineering logic
│   │   ├── __init__.py
│   │   └── pipelines.py
│   ├── features/         # (Optional) Separate feature definition/storage logic
│   │   └── __init__.py
│   ├── training/         # Model training, evaluation, tuning scripts
│   │   ├── __init__.py
│   │   ├── train_mlflow.py # Baseline training with MLflow
│   │   ├── tune_model.py   # Hyperparameter tuning script
│   │   └── train_smote.py  # Training with SMOTE script
│   ├── api/              # Flask API for serving predictions
│   │   ├── __init__.py
│   │   ├── app.py        # Flask app factory and entry point
│   │   ├── controllers/
│   │   │   ├── __init__.py
│   │   │   └── prediction_controller.py # Handles /predict route
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── prediction_service.py # Core prediction logic, model loading
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── prediction_schema.py # Pydantic schemas for API validation
│   ├── utils/            # Utility functions (e.g., config loading - conceptual)
│   │   └── __init__.py
│   └── main.py           # (Optional) Main script to run full pipelines
├── tests/                # Automated tests (conceptual structure)
│   ├── __init__.py
│   ├── unit/             # Unit tests for individual components
│   └── integration/      # Integration tests for component interactions
├── models/               # Saved model artifacts (usually outside Git, managed by MLflow/DVC)
├── k8s/                  # Kubernetes manifest files
│   ├── deployment.yaml   # Defines how to run the API pods
│   └── service.yaml      # Defines how to expose the API pods
├── Dockerfile            # Instructions to build the Docker container image
├── .dockerignore         # Files/directories ignored by Docker build
└── config.yaml           # (Optional) Project-wide configurations
```

**simple**

```
hvs_fire_prediction/
├── .git/
├── .github/
│   └── workflows/
│       └── cicd.yaml     # Conceptual CI/CD pipeline (GitHub Actions)
├── .gitignore
├── README.md             # This file
├── requirements.txt
├── notebooks/
│   └── 01_EDA_Scenario_Analysis.ipynb
├── data/
│   ├── raw/              # e.g., dummy_fault_RackA2_Module01_200rows.csv
│   └── processed/        # e.g., processed_dummy_fault_RackA2_Module01_200rows.csv
├── src/
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── simulator.py  # Original physics-based simulator (development version)
│   │   └── generate_dummy_data.py # Script used for generating data
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── pipelines.py    # Preprocessing and feature engineering logic
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_mlflow.py # Baseline training with MLflow
│   │   ├── tune_model.py   # Hyperparameter tuning with MLflow
│   │   └── train_smote.py  # SMOTE experiment with MLflow
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py        # Flask app factory and entry point
│   │   ├── controllers/
│   │   │   ├── __init__.py
│   │   │   └── prediction_controller.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── prediction_service.py
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── prediction_schema.py
│   └── utils/ # (Conceptual)
│       └── __init__.py
├── k8s/
│   ├── deployment.yaml   # Kubernetes Deployment manifest
│   └── service.yaml      # Kubernetes Service manifest
├── Dockerfile            # Instructions to build the Docker image
└── .dockerignore         # Files/directories ignored by Docker
```

## 3. Setup Instructions
**Prerequisites:**
* Python: Version 3.9+ recommended (check Dockerfile for specific version used in container).
* Git: For version control.
* Docker Desktop / Docker Engine: Required for building and running the container (Phase 7).
* (Optional) Local Kubernetes: Minikube, k3s, Kind, or similar for Phase 8 deployment testing.
* (Optional) kubectl: Kubernetes command-line tool.

**Steps:**
1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd hvs_fire_prediction
    ```
2.  **Create and Activate Virtual Environment:**
    * (Using PowerShell on Windows)
        ```powershell
        # Navigate to the project root directory
        cd path\to\hvs_fire_prediction

        # Create virtual environment
        python -m venv venv

        # Activate virtual environment
        .\venv\Scripts\activate
        ```
    * (Using bash on macOS/Linux)
        ```bash
        # Navigate to the project root directory
        cd path/to/hvs_fire_prediction

        # Create virtual environment
        python3 -m venv venv

        # Activate virtual environment
        source venv/bin/activate
        ```
    Your terminal prompt should now show `(venv)`.
3.  **Install Dependencies:**
    Install all required Python packages using the provided `requirements.txt` file.
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    (See `requirements.txt` section below for package list)

## 4. Project Workflow Phases
This project follows an iterative MLOps workflow:

**Phase 0: Project Setup & Foundational Understanding**
* **Goal:** Define scope, understand the domain (Li-ion thermal runaway), research approaches, set up the development environment, Git repository, and project structure.
* **Outcome:** Established project structure, initial tooling setup, clear objectives.

**Phase 1: Data Generation**
* **Goal:** Create time-series data representing normal and pre-failure battery states for 64 modules in a simulated warehouse.
* **Initial Approach:** Developed `src/data_generation/simulator.py` using simplified physics.
* **Challenge:** Reliably generating `Hazard_Label = 1` via physics simulation proved difficult during debugging.
* **Action (Pivot):** Created `src/data_generation/generate_dummy_data.py` to generate data with explicitly injected anomalies and hazardous labels for `RackA2_Module01` (approx. 200 rows) to ensure usable data for subsequent phases. Includes noise, missing values, and outliers.
* **To Run (Dummy Data):**
    ```bash
    python src/data_generation/generate_dummy_data.py
    ```
    (Output: e.g., `data/raw/dummy_fault_RackA2_Module01_200rows.csv`)

**Phase 2: Exploratory Data Analysis (EDA)**
* **Goal:** Understand the generated dummy data's characteristics, patterns, quality, and imbalance.
* **Tool:** Jupyter Notebook (`notebooks/01_EDA_Scenario_Analysis.ipynb`).
* **Activities:** Loaded data, checked structure/types/NaNs, analyzed target distribution (confirmed imbalance), visualized time-series for faulty vs. normal modules, used boxplots for outlier identification.
* **Key Findings:** Confirmed data validity, significant class imbalance, presence of imperfections, clear visual distinction for injected hazardous state.

**Phase 3: Preprocessing & Feature Engineering**
* **Goal:** Clean the raw dummy data and create additional informative features for modeling.
* **Implementation:** `src/preprocessing/pipelines.py`.
    * Handles timestamps, sorts data per module.
    * Imputes missing values (grouped ffill/bfill).
    * Handles outliers (quantile clipping per module).
    * Engineers delta features (rate of change, e.g., `Delta_1_Module_Avg_Surface_Temp_C`).
    * Engineers rolling window statistics (mean, std dev for 1, 5, 10 min windows, e.g., `Rolling_mean_6_Module_Avg_Surface_Temp_C`).
* **To Run:**
    ```bash
    # Ensure RAW_DATA_FILENAME in the script's main block matches dummy data file
    python src/preprocessing/pipelines.py
    ```
    (Output: e.g., `data/processed/processed_dummy_fault_RackA2_Module01_200rows.csv`)

**Phase 4: Baseline Model Training & Evaluation**
* **Goal:** Train initial classification models (Logistic Regression, Random Forest, XGBoost) and evaluate their baseline performance, applying class weighting to handle imbalance.
* **Implementation:** Logic initially developed in `src/training/train.py` (later integrated with MLflow).
* **Key Steps:** Load processed data, split into stratified train/test sets, scale features (StandardScaler), train models with `class_weight='balanced'` or `scale_pos_weight`, evaluate using Accuracy, Confusion Matrix, Classification Report (focusing on Precision/Recall/F1 for class 1), AUC-ROC, AUC-PR.
* **Outcome:** Initial performance benchmarks established for different model types.

**Phase 5: MLOps - Experiment Tracking (MLflow)**
* **Goal:** Systematically track experiments, parameters, metrics, and artifacts for reproducibility and comparison.
* **Implementation:** `src/training/train_mlflow.py` (integrates MLflow into baseline training), `src/training/tune_model.py` (integrates MLflow into tuning), `src/training/train_smote.py` (integrates MLflow into SMOTE experiment).
    * Uses `mlflow.start_run()`, `mlflow.log_param()`, `mlflow.log_metric()`, `mlflow.log_artifact()`, `mlflow.sklearn.log_model()`, `mlflow.xgboost.log_model()`.
    * Organizes runs into experiments (e.g., "Baseline", "Tuning", "SMOTE").
* **To Run (Example - Baseline with MLflow):**
    ```bash
    # Ensure PROCESSED_FILENAME in the script matches processed file
    python src/training/train_mlflow.py
    ```
* **Viewing Results:**
    ```bash
    mlflow ui
    ```
    (Access UI via browser at `http://127.0.0.1:5000`)

**Phase 5.1: Hyperparameter Tuning**
* **Goal:** Optimize the performance of the chosen model (e.g., XGBoost) by finding better hyperparameters.
* **Implementation:** `src/training/tune_model.py`.
    * Uses `RandomizedSearchCV` with `StratifiedShuffleSplit` cross-validation.
    * Defines a parameter distribution space (`PARAM_DISTRIBUTIONS`).
    * Optimizes for F1-score of the hazardous class (`SCORING_METRIC`).
    * Logs best parameters and final test set evaluation to MLflow (`HVS_Fire_Prediction_Tuning` experiment).
* **To Run:**
    ```bash
    # Ensure PROCESSED_FILENAME in the script matches processed file
    python src/training/tune_model.py
    ```

**Phase 5.2: Experimenting with SMOTE**
* **Goal:** Evaluate SMOTE as an alternative data-level approach to handle class imbalance.
* **Implementation:** `src/training/train_smote.py`.
    * Uses `imblearn.over_sampling.SMOTE`.
    * Applies SMOTE only to the scaled training data after splitting.
    * Trains XGBoost without class weighting on the SMOTE'd data.
    * Evaluates on the original scaled test set.
    * Logs results to MLflow (`HVS_Fire_Prediction_SMOTE` experiment) for comparison with class weighting approach.
* **To Run:**
    ```bash
    # Ensure PROCESSED_FILENAME in the script matches processed file
    python src/training/train_smote.py
    ```

**Phase 6: API Development (Flask)**
* **Goal:** Create a web service to serve predictions from the best trained model identified via MLflow experiments.
* **Implementation:** Located in `src/api/`.
    * `schemas/prediction_schema.py`: Pydantic models for request/response validation.
    * `services/prediction_service.py`: Loads model/scaler from MLflow (using `BEST_MLFLOW_RUN_ID` env var), preprocesses input, makes predictions. Contains core logic.
    * `controllers/prediction_controller.py`: Defines `/predict` endpoint (POST), handles requests, calls service, formats response using Flask Blueprint.
    * `app.py`: Creates Flask app, registers blueprint, configures basic logging, includes Prometheus instrumentation hook.
* **To Run Locally:**
    1.  Select best model's Run ID from MLflow UI.
    2.  Set environment variable:
        * PowerShell: `$env:BEST_MLFLOW_RUN_ID="YOUR_SELECTED_RUN_ID"`
        * bash: `export BEST_MLFLOW_RUN_ID="YOUR_SELECTED_RUN_ID"`
    3.  Run:
        ```bash
        python src/api/app.py
        ```
    API available at `http://127.0.0.1:5001`.

**Phase 7: Containerization (Docker)**
* **Goal:** Package the Flask API and dependencies into a portable Docker image.
* **Files:** `Dockerfile` (build instructions), `.dockerignore` (files to exclude).
* **To Build:**
    ```bash
    docker build -t your-image-name:tag .
    ```
* **To Run:**
    ```bash
    docker run -p 5001:5001 -e BEST_MLFLOW_RUN_ID="YOUR_RUN_ID" your-image-name:tag
    ```

**Phase 8: Deployment (Kubernetes)**
* **Goal:** Deploy the containerized API onto Kubernetes for scalability and high availability.
* **Files:** `k8s/deployment.yaml` (defines 3 replicas, image path, env vars), `k8s/service.yaml` (exposes deployment via NodePort).
* **Action:** Requires modifying manifests with actual image path and Run ID, then applying using `kubectl apply -f k8s/`. Requires a running K8s cluster (local or cloud).

**Phase 9: Monitoring (Prometheus & Grafana)**
* **Goal:** Observe API performance and health.
* **Concept:** Instrument API with `prometheus-flask-exporter` (`/metrics` endpoint), deploy Prometheus to scrape metrics, deploy Grafana to visualize metrics (latency, throughput, errors, prediction distribution) using PromQL queries. Requires deploying and configuring Prometheus/Grafana.

**Phase 10: CI/CD Automation (GitHub Actions)**
* **Goal:** Automate testing, building, and deployment on code changes.
* **File:** `.github/workflows/cicd.yaml` (conceptual definition).
* **Workflow:** Trigger (push/PR) -> Setup -> Lint -> Test -> Build Image -> Push Image -> Deploy to K8s (`kubectl apply`). Requires secrets configuration in GitHub.

**Phase 11: Data Drift & Retraining Strategy**
* **Goal:** Maintain model performance over time.
* **Concept:**
    * **Drift Detection:** Monitor input feature distributions (KS-test, PSI) and prediction probabilities against a baseline.
    * **Retraining Triggers:** Significant drift, performance degradation below threshold (e.g., low Recall), scheduled intervals.
    * **Automated Pipeline:** Trigger -> Collect/Label Data -> Preprocess -> Retrain -> Validate -> Deploy (if better). Requires orchestration tools and robust labeling.

**Phase 12: Documentation**
* **Goal:** Ensure project is understandable and maintainable.
* **Components:** This `README.md`, code comments/docstrings, API documentation (schemas), MLflow experiment summaries, conceptual deployment/operations guide.

## 5. Requirements (`requirements.txt`)
```
**Core Data Handling & ML**
pandas>=1.5,<3.0
numpy>=1.21,<2.0
scikit-learn>=1.1,<1.5
xgboost>=1.5,<2.1
scipy>=1.8,<2.0 # For tuning distributions

**Imbalance Handling (if using SMOTE script)**
imbalanced-learn>=0.10,<0.13

**API Framework & Validation**
Flask>=2.2,<3.1
pydantic>=2.0,<3.0 # Assuming Pydantic v2+ for .model_dump()

**Monitoring (API Instrumentation)**
prometheus-flask-exporter>=0.18,<0.23

**MLflow (Tracking & Model Loading)**
mlflow>=2.0,<2.14

**Model/Scaler Saving (if not solely relying on MLflow format)**
joblib>=1.1,<1.5

WSGI Server (Recommended for Production instead of Flask dev server)
gunicorn>=20.0,<22.0 # Example, use if deploying with Gunicorn
waitress>=2.0,<3.0 # Example, alternative WSGI server for Windows/Linux

**Note: Pinning versions (like pandas==2.1.4) ensures greater reproducibility**
than using ranges (>=, <). Ranges are used here for broader compatibility
during development. For production, pin exact versions tested.
```

## 6. Future Work / Improvements
* Implement physics-based simulator (`simulator.py`) reliably.
* Write comprehensive unit and integration tests.
* Implement robust data labeling strategy for retraining.
* Fully implement CI/CD deployment to a target K8s cluster with proper secret management.
* Deploy and configure Prometheus/Grafana stack for live monitoring.
* Implement automated drift detection checks using tools like Evidently AI or NannyML.
* Build a full retraining pipeline using an orchestrator (Airflow, Kubeflow Pipelines).
* Add more sophisticated feature engineering based on domain expertise.
* Explore different model architectures (e.g., LSTMs, Transformers for time-series).
* Use Kubernetes Secrets/ConfigMaps for configuration instead of hardcoding/env vars.
* Use a production-grade WSGI server (Gunicorn/Waitress) in the Docker container.
* Implement more sophisticated outlier detection methods.
```
