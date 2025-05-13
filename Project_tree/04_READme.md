
---

### Phase 4: Baseline Model Development, Training & Evaluation

**1. Goal of this Phase:**
To train several standard classification machine learning models on the preprocessed and feature-engineered data, evaluate their initial (baseline) performance without extensive hyperparameter tuning, and establish benchmarks. A key focus is to incorporate strategies from the outset to handle the significant class imbalance identified in the EDA phase.

**2. Rationale (Why this phase is critical):**
* **Establish Performance Baselines:** Before investing time in complex model architectures or exhaustive hyperparameter searches, it's crucial to understand how well standard, off-the-shelf models perform. This provides a reference point to measure the effectiveness of more advanced techniques later. If a simple model performs surprisingly well, it might question the need for excessive complexity.
* **Validate Data & Features:** This phase serves as an initial validation of whether the preprocessed data and engineered features actually contain enough predictive signal for the models to learn the difference between normal and hazardous states.
* **Initial Imbalance Handling:** Given the known class imbalance (very few hazardous samples), it's important to apply basic techniques to address this during the first modeling attempts. Ignoring it would likely lead to models that perform very poorly on the minority (hazardous) class.
* **Model Candidate Identification:** The performance of these baseline models helps in identifying one or two promising model types (e.g., linear models, tree-based ensembles, gradient boosting) that could be further optimized in subsequent phases.
* **Early Proof-of-Concept:** This phase demonstrates whether the overall approach (data -> features -> model -> prediction) is viable for the problem at hand.

**3. Approach & Key Activities:**
The logic for this phase was initially developed as a Python script (conceptually `src/training/train.py`) and later evolved into `src/training/train_mlflow.py` when MLflow integration was added in Phase 5. For this phase description, we focus on the core training and evaluation steps.

* **Load Processed Data:**
    * The script begins by loading the processed dataset from Phase 3 (e.g., `data/processed/processed_dummy_fault_RackA2_Module01_200rows.csv`).
    * **Code Snippet (Conceptual):**
        ```python
        import pandas as pd
        df_processed = pd.read_csv("data/processed/processed_dummy_fault_RackA2_Module01_200rows.csv")
        ```

* **Feature and Target Selection (X, y):**
    * The `Hazard_Label` column was designated as the target variable `y`.
    * All other relevant numerical columns (which include the original sensor readings that were kept, plus all the newly engineered delta and rolling features) were selected as input features `X`.
    * Identifier columns like `Timestamp`, `Module_ID`, and `Rack_ID` were excluded from the feature set `X` as they are not direct inputs for the predictive model algorithm (though `Module_ID` could be used for grouped cross-validation, which is a more advanced technique not covered in this baseline).
    * **Code Snippet (Conceptual):**
        ```python
        TARGET_COLUMN = 'Hazard_Label'
        EXCLUDE_COLS = ['Timestamp', 'Module_ID', 'Rack_ID', TARGET_COLUMN]
        y = df_processed[TARGET_COLUMN]
        features = [col for col in df_processed.select_dtypes(include=np.number).columns if col not in EXCLUDE_COLS]
        X = df_processed[features]
        ```

* **Train/Test Split:**
    * The dataset (`X` and `y`) was split into a training set (e.g., 80% of the data) and a testing set (e.g., 20%).
    * **Stratified Splitting:** `StratifiedShuffleSplit` from `scikit-learn` (or `train_test_split` with the `stratify=y` argument) was crucial here. Stratification ensures that the proportion of samples from each class (0 and 1) is maintained in both the training and testing subsets. This is vital for imbalanced datasets to ensure that the rare class is adequately represented in both sets for reliable training and evaluation.
    * **Code Snippet:**
        ```python
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        ```

* **Data Scaling (Feature Scaling):**
    * A `StandardScaler` from `scikit-learn` was used to standardize the features (transforming them to have zero mean and unit variance).
    * **Fit on Training Data Only:** The scaler was **fitted only on the training data (`X_train`)**. This means it learned the mean and standard deviation from the training data alone.
    * **Transform Both Sets:** The *same* fitted scaler was then used to **transform both the training data (`X_train`) and the test data (`X_test`)**. This prevents data leakage from the test set into the training process, which would lead to overly optimistic and unrealistic performance estimates.
    * **Rationale:** Scaling is important for many ML algorithms, especially those that are distance-based (like SVMs, KNNs) or rely on gradient descent (like Logistic Regression, Neural Networks). While tree-based models (Random Forest, XGBoost) are less sensitive to feature scaling, applying it consistently doesn't hurt and can sometimes help even them.
    * **Code Snippet:**
        ```python
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # The 'scaler' object would be saved (e.g., using joblib) to apply the same transformation to new data during inference.
        ```

* **Imbalance Handling Strategy (Class Weighting):**
    * As identified in EDA, our dataset is highly imbalanced. For this baseline phase, **class weighting** was the chosen strategy.
    * **Logistic Regression & Random Forest:** The `class_weight='balanced'` parameter was used. This mode automatically adjusts the weights assigned to each class in the model's loss function, making them inversely proportional to class frequencies. This means the model gets penalized more for misclassifying the rare hazardous class.
    * **XGBoost:** The `scale_pos_weight` parameter was used. This is calculated as `count(negative_class_samples_in_train) / count(positive_class_samples_in_train)`. For our data, this value would be very high (e.g., ~700 if 0.145% are positive), telling XGBoost to give much more importance to correctly classifying the positive (hazardous) samples.
    * **Rationale:** Applying class weighting at the algorithm level is a relatively straightforward way to mitigate the impact of class imbalance without altering the dataset itself through resampling (like SMOTE, which would be explored later).

* **Model Training:**
    * Three common and effective classification models were instantiated and trained:
        1.  **`LogisticRegression`:** A simple, interpretable linear model. Good as a basic baseline. Used with `solver='liblinear'` and `class_weight='balanced'`.
        2.  **`RandomForestClassifier`:** An ensemble model using multiple decision trees. Generally robust and performs well on various tasks. Used with `n_estimators=100` (a common default) and `class_weight='balanced'`.
        3.  **`XGBClassifier` (XGBoost):** A powerful and popular gradient boosting algorithm known for high performance. Used with `objective='binary:logistic'`, `eval_metric='logloss'`, and the calculated `scale_pos_weight`.
    * Each model was trained using its `.fit()` method on the scaled training data (`X_train_scaled`, `y_train`).

* **Prediction & Evaluation:**
    * After training, each model was used to make predictions on the unseen, scaled *test data* (`X_test_scaled`).
    * `model.predict(X_test_scaled)`: To get the binary class predictions (0 or 1).
    * `model.predict_proba(X_test_scaled)[:, 1]`: To get the probability scores for the positive class (label 1). These probabilities are needed for AUC-ROC and AUC-PR calculations.
    * **Key Evaluation Metrics** (calculated using `sklearn.metrics`):
        * **Accuracy Score:** `accuracy_score(y_test, y_pred)` - Percentage of correct predictions. Not very informative for imbalanced data.
        * **Confusion Matrix:** `confusion_matrix(y_test, y_pred)` - A table showing TP, TN, FP, FN. Visualized using `ConfusionMatrixDisplay`.
        * **Classification Report:** `classification_report(y_test, y_pred)` - Provides:
            * **Precision:** For class 1, it's `TP / (TP + FP)` (out of all predicted hazardous, how many were actually hazardous?).
            * **Recall (Sensitivity):** For class 1, it's `TP / (TP + FN)` (out of all actual hazardous, how many did we find?). **This is a primary metric for us.**
            * **F1-Score:** Harmonic mean of Precision and Recall (`2 * (Precision * Recall) / (Precision + Recall)`).
        * **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** `roc_auc_score(y_test, y_pred_proba)` - Measures the model's ability to distinguish between classes across various thresholds.
        * **AUC-PR (Area Under the Precision-Recall Curve):** Calculated using `precision_recall_curve` and `auc(recall, precision)`. This is generally a more informative metric than AUC-ROC for highly imbalanced datasets because it focuses on the performance of the minority class.

**4. Technical Insights & Decisions:**
* **Model Choices:** Logistic Regression (simple baseline), Random Forest (robust ensemble), and XGBoost (high-performance gradient boosting) provide a good spectrum of model complexity and capabilities.
* **Prioritizing Recall and AUC-PR:** Given the problem context (predicting hazardous events where misses are costly), Recall for the hazardous class (label 1) and AUC-PR were considered more important than overall accuracy. The goal was to achieve high recall while keeping precision at an acceptable level (to avoid too many false alarms).
* **Iterative Approach to Imbalance:** Class weighting was chosen as the first method to tackle imbalance. If results were insufficient, further techniques like SMOTE would be considered in later experimental iterations (which we did in Phase 5.2).
* **Reproducibility:** Using `random_state` in data splitting and model instantiation ensures that the results are reproducible if the script is run again with the same data and settings.

**5. Key Commands/Code Snippets (Illustrative from `train_mlflow.py`):**
* **Feature/Target Split:**
    ```python
    y = df[TARGET_COLUMN]
    X = df[features_list]
    ```
* **Stratified Train/Test Split:**
    ```python
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        # ... (y_train, y_test)
    ```
* **Scaling:**
    ```python
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```
* **XGBoost with `scale_pos_weight`:**
    ```python
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1
    model = XGBClassifier(scale_pos_weight=scale_pos_weight_val, ...)
    model.fit(X_train_scaled, y_train)
    ```
* **Getting Metrics:**
    ```python
    from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    # ... (calculate metrics) ...
    ```

**6. Outcome of Phase 4:**
* Baseline performance metrics (Precision, Recall, F1-score for class 1; AUC-ROC, AUC-PR) for three different classification models (Logistic Regression, Random Forest, XGBoost), all trained using class weighting to address imbalance.
* A foundational understanding of how well standard algorithms perform on the processed and feature-engineered data.
* Identification of the most promising model type(s) for further optimization (e.g., hyperparameter tuning) and experimentation (e.g., trying SMOTE).
* A Python script (`src/training/train.py`, later evolved into `src/training/train_mlflow.py`) for reproducible model training and evaluation.

This phase provided the first real test of our data's predictive quality and set the stage for more focused model improvement efforts.

---
