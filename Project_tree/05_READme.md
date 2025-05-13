
---

### Phase 5: MLOps - Experiment Tracking (MLflow) & Iterative Model Improvement

**1. Goal of this Phase:**
To introduce systematic experiment tracking using MLflow and explore techniques to improve upon the baseline model performance. This involved:
* **MLflow Integration:** Incorporating MLflow into the training pipeline to log parameters, metrics, and model artifacts for reproducibility and comparison.
* **Hyperparameter Tuning:** Optimizing the hyperparameters of the most promising baseline model (e.g., XGBoost) to enhance its predictive performance.
* **Alternative Imbalance Handling (SMOTE):** Experimenting with SMOTE as an alternative or complementary technique to class weighting for addressing the severe class imbalance.

**2. Rationale (Why this phase is critical):**
* **Scientific Rigor & Reproducibility (MLflow):** As we start experimenting with different model settings, features, or techniques, it's vital to keep a record of what was done and what the outcome was. MLflow provides this capability, preventing a chaotic and untraceable experimentation process. It's a cornerstone of MLOps.
* **Performance Optimization (Hyperparameter Tuning):** Default hyperparameters are rarely optimal for a specific dataset. Tuning allows us to find a configuration that maximizes the model's ability to learn the desired patterns, potentially leading to significant gains in key metrics like Recall and Precision for the hazardous class.
* **Addressing Imbalance Robustly (SMOTE):** While class weighting is a good start, data-level techniques like SMOTE can sometimes provide a different kind of boost to model performance on the minority class by creating more diverse (synthetic) examples for the model to learn from. Experimenting with it helps determine the best strategy for our specific problem.
* **Informed Model Selection:** By logging all experiments in MLflow, we can make data-driven decisions about which model version and which techniques yield the best results according to our project's success criteria.

**3. Approach & Key Activities:**

This phase involved evolving our training scripts to incorporate MLflow and then creating specialized scripts for tuning and SMOTE experiments.

* **A. MLflow Integration into Baseline Training (`src/training/train_mlflow.py` - ID: `training_script_mlflow_v1`):**
    * **Objective:** Modify the Phase 4 baseline training script to log all relevant information for each model (Logistic Regression, Random Forest, XGBoost trained with class weighting) to MLflow.
    * **Key Activities:**
        1.  **Import `mlflow`:** Added `import mlflow`, `mlflow.sklearn`, `mlflow.xgboost`.
        2.  **Set Experiment:** Used `mlflow.set_experiment("HVS_Fire_Prediction_Baseline")` to group these baseline runs.
        3.  **`mlflow.start_run()`:** Wrapped the training and evaluation for each model type within a `with mlflow.start_run(run_name=model_name, nested=True):` block. Nested runs help organize experiments if, for example, the data scaling step was logged as a parent run.
        4.  **Log Parameters (`mlflow.log_params()`):** Logged key model parameters (e.g., `solver`, `class_weight` for Logistic Regression; `n_estimators`, `scale_pos_weight` for XGBoost) and general parameters like number of features used.
        5.  **Log Metrics (`mlflow.log_metric()`):** Logged all key evaluation metrics: accuracy, precision/recall/f1 for class 1, macro precision/recall/f1, AUC-ROC, AUC-PR, and individual components of the confusion matrix (TP, TN, FP, FN).
        6.  **Log Artifacts (`mlflow.log_artifact()`):** Saved and logged the confusion matrix plot (`.png`) and the full classification report (`.txt`) as artifacts for each run.
        7.  **Log Model (`mlflow.sklearn.log_model()`, `mlflow.xgboost.log_model()`):** Used MLflow's model flavors to log the trained model object itself, including its dependencies and an input example (a small slice of the training data). This makes model deployment and reloading much easier.
        8.  **Log Scaler:** The `StandardScaler` object, fitted on the training data, was also saved (e.g., using `joblib`) and logged as an artifact, as it's essential for preprocessing new data during inference.
    * **Running:** `python src/training/train_mlflow.py`
    * **Outcome:** All baseline model runs were logged to the "HVS\_Fire\_Prediction\_Baseline" experiment in MLflow, providing a clear, versioned record.

* **B. Hyperparameter Tuning (e.g., for XGBoost) with MLflow (`src/training/tune_model.py` - ID: `tuning_script_v1`):**
    * **Objective:** To find an optimal set of hyperparameters for the XGBoost model (it was a promising candidate from baseline) to improve its performance on the hazardous class.
    * **Key Activities:**
        1.  **Select Model:** Chose XGBoost for tuning.
        2.  **Define Parameter Search Space (`PARAM_DISTRIBUTIONS`):** Specified a dictionary of XGBoost hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`) and the distributions/ranges to sample from (e.g., `scipy.stats.randint`, `scipy.stats.uniform`).
        3.  **Choose Scoring Metric (`SCORING_METRIC`):** Defined the metric to optimize during the search. `make_scorer(f1_score, pos_label=1)` was used to focus on the F1-score of the hazardous class (label 1), providing a balance between precision and recall for this critical class.
        4.  **`RandomizedSearchCV`:** Used `sklearn.model_selection.RandomizedSearchCV` with `StratifiedShuffleSplit` for cross-validation.
            * `n_iter`: Number of different parameter combinations to try (e.g., 20).
            * `cv`: Number of cross-validation folds (e.g., 3).
        5.  **MLflow Integration:**
            * A new experiment, e.g., `"HVS_Fire_Prediction_Tuning"`, was set.
            * A main MLflow run was started for the entire tuning process.
            * `RandomizedSearchCV` itself doesn't automatically log each trial to MLflow. To achieve this (for very detailed tracking), one would typically wrap the `fit` call or use callbacks, or manually iterate and log. For this script, we focused on logging the *best* parameters and results found by `RandomizedSearchCV`.
            * **Logged Information:** Best parameters found (`random_search.best_params_`), best cross-validation score (`random_search.best_score_`), and the evaluation metrics of the final model (trained with best params) on the test set. The final tuned model and the scaler were also logged as artifacts.
    * **Running:** `python src/training/tune_model.py`
    * **Outcome:** Identification of a potentially better set of hyperparameters for XGBoost, with all results logged in MLflow for comparison against the baseline.

* **C. Experimenting with SMOTE with MLflow (`src/training/train_smote.py` - ID: `training_script_smote_v1`):**
    * **Objective:** Evaluate SMOTE as an alternative to class weighting for handling imbalance with XGBoost.
    * **Key Activities:**
        1.  **Import SMOTE:** From `imblearn.over_sampling import SMOTE`.
        2.  **Data Preparation:**
            * Loaded processed data, performed train/test split (stratified).
            * **Scaled features:** Fitted `StandardScaler` on original `X_train`, then transformed `X_train` and `X_test`.
            * **Applied SMOTE:** Initialized `SMOTE(random_state=RANDOM_STATE)` and applied it **only to the scaled training data**: `X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)`. The test set (`X_test_scaled`, `y_test`) was **not** modified by SMOTE.
        3.  **Model Training:** Trained an `XGBClassifier` **without** `scale_pos_weight` (since SMOTE aims to balance the class distribution of the training data directly).
        4.  **Evaluation:** Evaluated the model on the original, scaled, *un-SMOTE'd* test set (`X_test_scaled`, `y_test`).
        5.  **MLflow Integration:**
            * Logged to a new experiment (e.g., `"HVS_Fire_Prediction_SMOTE"`).
            * Logged parameters indicating SMOTE was used (e.g., `imbalance_technique: 'SMOTE'`, `smote_k_neighbors`).
            * Logged metrics, model, and scaler as in other experiments.
    * **Running:** `python src/training/train_smote.py` (after `pip install imbalanced-learn`).
    * **Outcome:** Performance metrics for XGBoost trained on SMOTE'd data, logged in MLflow, allowing direct comparison with the class-weighting approach.

**4. Technical Insights & Decisions:**
* **Why MLflow Early?** Introducing MLflow before extensive tuning or trying many different preprocessing steps is crucial. It establishes a disciplined approach to experimentation from the start.
* **Choosing a Tuning Strategy (`RandomizedSearchCV`):** For a potentially large hyperparameter space, `RandomizedSearchCV` is often more efficient than `GridSearchCV` as it doesn't try every single combination but samples randomly. It's a good balance between search breadth and computational cost.
* **Scoring Metric for Tuning:** Optimizing for `f1_score` (with `pos_label=1`) during tuning for an imbalanced problem is generally better than optimizing for plain accuracy. It seeks a balance between precision and recall for the important minority class.
* **Correct Application of SMOTE:** The most critical aspect of using SMOTE (or any resampling) is applying it *only* to the training data *after* the train/test split and *before* model training. The scaler should also be fitted on the original training data before SMOTE. This prevents data leakage and ensures a realistic evaluation on the untouched test set.
* **Comparing Imbalance Techniques:** MLflow allowed us to compare the effectiveness of class weighting (used in baseline and tuning) versus SMOTE by looking at their respective logged metrics (especially Recall, Precision for class 1, and AUC-PR). The "best" technique can be data and model-dependent.

**5. Key Commands/Code Snippets:**
* **Installing Libraries:**
    ```bash
    pip install mlflow joblib xgboost scikit-learn pandas numpy scipy imbalanced-learn
    ```
* **Starting MLflow UI (from project root):**
    ```bash
    mlflow ui
    ```
* **MLflow Basic Logging (Conceptual):**
    ```python
    import mlflow
    mlflow.set_experiment("My_Experiment")
    with mlflow.start_run(run_name="My_Run"):
        mlflow.log_param("my_param", 5)
        # ... train model ...
        mlflow.log_metric("recall_class1", 0.92)
        mlflow.sklearn.log_model(sk_model=my_model, artifact_path="my_sklearn_model")
    ```
* **RandomizedSearchCV (Conceptual):**
    ```python
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    param_dist = {'learning_rate': uniform(0.01, 0.2), 'n_estimators': [100, 200, 300]}
    random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=10, cv=3, scoring='f1_weighted')
    random_search.fit(X_train_scaled, y_train)
    best_model = random_search.best_estimator_
    ```
* **SMOTE (Conceptual):**
    ```python
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train)
    # ... then train model on X_train_smote, y_train_smote ...
    ```

**6. Outcome of Phase 5:**
* **Organized Experiments:** All training runs (baseline, tuning trials, SMOTE experiments) are logged in MLflow with their parameters, metrics, and artifacts.
* **Reproducibility:** Any specific model run can be revisited and potentially reproduced.
* **Model Comparison:** The MLflow UI provides a platform to easily compare the performance of different models and techniques (e.g., class weighting vs. SMOTE, different hyperparameter sets).
* **Selection of "Best" Model:** Based on the logged metrics (e.g., highest Recall for class 1 while maintaining acceptable Precision, or best PR-AUC), a "best" performing model and its associated Run ID can be identified for deployment in the next phase (API).
* **Key MLOps Practice Implemented:** This phase directly fulfills the project's requirement for using MLflow for experiment tracking and versioning.

This iterative improvement and tracking phase is central to developing a high-performing and reliable machine learning model.

---
