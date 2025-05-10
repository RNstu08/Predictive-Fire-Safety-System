**Phase 2: Exploratory Data Analysis (EDA)**.

**1. Goal of this Phase:**
To thoroughly examine, visualize, and understand the synthetic dummy dataset (e.g., dummy_fault_RackA2_Module01_200rows.csv) generated in Phase 1. The specific objectives were to:

Validate the structure and content of the dataset against our expectations (e.g., correct number of modules, features, presence of hazardous labels).
Identify and understand the characteristics of "Normal" operational data versus the injected "Hazardous" states for the faulty module.
Quantify the extent of class imbalance between normal and hazardous labels.
Observe and confirm the presence and nature of simulated data imperfections (noise, missing values, outliers).
Gather insights that would inform the subsequent data preprocessing steps and feature engineering strategies in Phase 3.

**2. Rationale (Why?) - Continued:**
EDA is a fundamental first step after acquiring or generating data in any data science or machine learning project. It's about "getting to know your data" before attempting to clean it or build models.

Data Understanding: Without EDA, you are working blind. You need to understand the distributions of your features, their typical ranges, and how they behave over time, especially in relation to the target variable.
Validate Assumptions & Data Generation: EDA allows us to confirm if the data generation (even dummy generation) worked as intended. Did we get the expected number of hazardous samples? Do the "faulty" sensor readings look significantly different from normal ones?
Identify Problems Early: It helps catch potential issues early on, such as incorrect data types, unexpected or excessive missing values, or if the hazardous signal isn't actually distinct enough in the features we've generated.
Guide Preprocessing: The nature and extent of missing values or outliers observed during EDA directly determine which imputation or outlier handling strategies will be most appropriate in the next phase.
Inform Feature Engineering: Understanding the raw features and their relationship with the target can spark ideas for creating new, more powerful features that might better capture the pre-failure signals.
Class Imbalance Assessment: Quantifying the imbalance between normal and hazardous states is crucial. This knowledge directly influences the choice of machine learning algorithms, evaluation metrics (accuracy becomes misleading), and specific techniques to handle the imbalance during model training.

**3. Approach & Key Activities:**
This phase was primarily conducted using a Jupyter Notebook (`notebooks/01_EDA_Scenario_Analysis.ipynb`).
* **Loading & Initial Inspection:**
    * The generated dummy CSV file (e.g., `dummy_fault_RackA2_Module01_200rows.csv`) was loaded into a Pandas DataFrame.
    * Basic DataFrame methods were used to understand its structure:
        * `df.head()` and `df.tail()`: To view the first and last few rows and get a feel for the data.
        * `df.info()`: To check column data types (e.g., ensuring 'Timestamp' is datetime, sensor readings are numeric), identify non-null counts per column (to spot missing data).
        * `df.describe()`: To get summary statistics (mean, std, min, max, quartiles) for all numerical features, helping understand typical ranges and potential extreme values.
        * `df.isnull().sum()`: To quantify missing values per column (confirming our simulated imperfections).
        * `df['Module_ID'].nunique()`: To verify the number of unique modules in the dataset (should be 64).
* **Target Variable Analysis (`Hazard_Label`):**
    * `df['Hazard_Label'].value_counts(normalize=True)`: Calculated the percentage distribution of the 'Normal' (0) and 'Hazardous' (1) classes.
    * Visualized this distribution using a bar plot (`sns.countplot` or Pandas `plot(kind='bar')`) to clearly show the class imbalance.
* **Time-Series Visualization:** This was the core of the EDA for this dataset.
    * **Isolating Data:** Data for the designated `FAILING_MODULE_ID` (e.g., "RackA2\_Module01") and a sample `sample_normal_module_id` was extracted into separate DataFrames for focused analysis. `Timestamp` was converted to datetime objects.
    * **Plotting Failing Module Data:** For the failing module, key sensor readings and engineered features (e.g., `Module_Avg_Surface_Temp_C`, `Module_Voltage_V`, `Sim_OffGas_CO_ppm_proxy`, `Module_SoC_percent`, and eventually the engineered `Delta_...` and `Rolling_...` features if EDA was revisited after Phase 3) were plotted against `Timestamp` using line plots (`sns.lineplot` or Pandas `plot()`).
        * Vertical lines or shaded regions were conceptually added to these plots to indicate the window where `Hazard_Label == 1`.
    * **Comparing Failing vs. Normal:** A key feature like `Module_Avg_Surface_Temp_C` was plotted for both the failing and a normal module on the same graph to visually highlight the divergence in behavior.
* **Data Quality Deep Dive (Outliers & Noise):**
    * **Visual Inspection of Time-Series Plots:** The line plots from the previous step were examined for jaggedness (indicating noise) and sudden, isolated spikes (indicating potential outliers separate from the main fault trend).
    * **Boxplots (`sns.boxplot`):**
        * A boxplot for a key feature like `Module_Avg_Surface_Temp_C` across *all* data points to see the overall distribution and how the fault condition temperatures appear as outliers.
        * Boxplots for several key features *just for the failing module* to understand its value distribution throughout its lifecycle (normal and faulty periods).

**4. Technical Insights & Decisions:**
* **Confirmation of Dummy Data Utility:** EDA confirmed that the dummy data script successfully generated a dataset with the intended structure, including the crucial `Hazard_Label = 1` instances and clear (albeit artificial) anomalous sensor readings for the faulty module during the hazardous window.
* **Verification of Imperfections:** The presence of missing values, noise (visible in line plots), and outliers (visible in boxplots) was confirmed, validating that the data would provide a good basis for practicing preprocessing techniques.
* **Class Imbalance Quantified:** The severe class imbalance (e.g., ~0.14% for label 1) was clearly identified. This immediately signaled that standard accuracy would be a poor evaluation metric for models and that techniques to handle imbalance (like class weighting or resampling) would be essential in the modeling phase.
* **Visual Predictive Signal:** The time-series plots for the failing module showed a clear visual distinction in sensor behavior (temperature rise, voltage/SoC drop, gas proxy increase) during the period we labeled as hazardous. This provided initial confidence that the features contained predictive signals.
* **Choice of EDA Tools:** Jupyter Notebooks were chosen for their interactivity, allowing for iterative exploration, code execution, and immediate visualization, which is ideal for EDA. Pandas, Matplotlib, and Seaborn are standard Python libraries for this purpose.

**5. Key Commands/Code Snippets (Illustrative from EDA Notebook):**
* **Loading Data:**
    ```python
    import pandas as pd
    df = pd.read_csv("../data/raw/dummy_fault_RackA2_Module01_200rows.csv")
    ```
* **Basic Info:**
    ```python
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    ```
* **Target Distribution:**
    ```python
    print(df['Hazard_Label'].value_counts(normalize=True))
    # sns.countplot(x='Hazard_Label', data=df) # Using Seaborn
    df['Hazard_Label'].value_counts().plot(kind='bar', title='Hazard Label Distribution') # Using Pandas
    plt.show()
    ```
* **Time-Series Plot (Conceptual for one feature):**
    ```python
    import matplotlib.pyplot as plt
    df_failing = df[df['Module_ID'] == 'RackA2_Module01'].copy()
    df_failing['Timestamp'] = pd.to_datetime(df_failing['Timestamp'])
    plt.figure(figsize=(12, 6))
    plt.plot(df_failing['Timestamp'], df_failing['Module_Avg_Surface_Temp_C'], label='Failing Module Temp')
    # Add logic here to highlight hazardous window, e.g., with axvspan or axvline
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()
    ```

**6. Outcome:**
* A thorough understanding of the structure, content, and quality of the generated dataset.
* Confirmation that the dummy data provides a suitable basis for demonstrating preprocessing and modeling, especially for an imbalanced classification problem.
* Clear identification of the class imbalance, which will inform modeling strategies.
* Visual confirmation that the features associated with the "hazardous" state are distinct.
* A Jupyter Notebook (`notebooks/01_EDA_Scenario_Analysis.ipynb`) documenting the EDA process and findings.

This comprehensive EDA ensures we are not proceeding with flawed or misunderstood data, which is a critical checkpoint before investing time in preprocessing and model building.

---

