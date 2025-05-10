# src/preprocessing/pipelines.py

import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Define default columns expected from the raw data generator
# (Adjust if your dummy generator script produces different columns)
EXPECTED_COLUMNS = [
    'Timestamp', 'Module_ID', 'Rack_ID', 'Module_Avg_Surface_Temp_C',
    'Module_Max_Surface_Temp_C', 'Module_Voltage_V', 'Module_Current_A',
    'Module_SoC_percent', 'Sim_OffGas_CO_ppm_proxy', 'Sim_OffGas_H2_ppm_proxy',
    'Sim_OffGas_VOC_ppm_proxy', 'Ambient_Temp_Rack_C', 'Hazard_Label'
]

# Columns that represent sensor readings and might need imputation/outlier handling/feature engineering
SENSOR_COLUMNS = [
    'Module_Avg_Surface_Temp_C', 'Module_Max_Surface_Temp_C', 'Module_Voltage_V',
    'Module_Current_A', 'Module_SoC_percent', 'Sim_OffGas_CO_ppm_proxy',
    'Sim_OffGas_H2_ppm_proxy', 'Sim_OffGas_VOC_ppm_proxy', 'Ambient_Temp_Rack_C'
]

# --- Preprocessing Functions ---

def load_data(file_path):
    """Loads data from a CSV file."""
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded data shape: {df.shape}")
    # Basic validation
    if not all(col in df.columns for col in EXPECTED_COLUMNS):
         print("Warning: Loaded data is missing expected columns.")
         # Optionally raise an error or list missing/extra columns
    return df

def handle_timestamp(df):
    """Converts Timestamp column to datetime and sets as index."""
    if 'Timestamp' not in df.columns:
        raise ValueError("DataFrame must contain a 'Timestamp' column.")
    
    print("Handling Timestamp...")
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        print("  Converted 'Timestamp' column to datetime objects.")
    
    # Sort by Timestamp (important for time-series operations like ffill)
    # We need to sort within each module group if data isn't already ordered
    if 'Module_ID' in df.columns:
        print("  Sorting data by Module_ID and Timestamp...")
        df = df.sort_values(by=['Module_ID', 'Timestamp'])
    else:
        print("  Sorting data by Timestamp...")
        df = df.sort_values(by='Timestamp')

    # Set Timestamp as index
    # df.set_index('Timestamp', inplace=True) # Setting index can complicate grouping later, let's keep as column for now
    # print("  Set 'Timestamp' as DataFrame index.")
    print("  Timestamp handling complete.")
    return df

def impute_missing_values(df, columns_to_impute, method='ffill'):
    """
    Handles missing values (NaNs) in specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_to_impute (list): List of column names to impute.
        method (str): Imputation method.
                      'ffill': Forward fill within each group (module).
                      'mean': Fill with the mean of each group (module).
                      'median': Fill with the median of each group (module).
                      (Add 'bfill' or others if needed)

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    print(f"Imputing missing values using method: '{method}'...")
    original_nan_counts = df[columns_to_impute].isnull().sum()

    if 'Module_ID' not in df.columns:
        print("  Warning: 'Module_ID' not found. Imputing globally instead of per module.")
        if method == 'ffill':
            df[columns_to_impute] = df[columns_to_impute].ffill()
            df[columns_to_impute] = df[columns_to_impute].bfill() # Backfill remaining NaNs at the start
        elif method == 'mean':
            for col in columns_to_impute:
                df[col] = df[col].fillna(df[col].mean())
        elif method == 'median':
            for col in columns_to_impute:
                df[col] = df[col].fillna(df[col].median())
        else:
            raise ValueError(f"Unsupported imputation method: {method}")
    else:
        print("  Imputing missing values per Module_ID group...")
        if method == 'ffill':
            # Forward fill within each module's time series
            df[columns_to_impute] = df.groupby('Module_ID')[columns_to_impute].ffill()
            # Backfill any remaining NaNs at the beginning of a module's series
            df[columns_to_impute] = df.groupby('Module_ID')[columns_to_impute].bfill()
        elif method == 'mean':
            # Fill with the mean *of that specific module*
            df[columns_to_impute] = df.groupby('Module_ID')[columns_to_impute].transform(lambda x: x.fillna(x.mean()))
        elif method == 'median':
            # Fill with the median *of that specific module*
            df[columns_to_impute] = df.groupby('Module_ID')[columns_to_impute].transform(lambda x: x.fillna(x.median()))
        else:
            raise ValueError(f"Unsupported imputation method: {method}")

    # Final check: Fill any remaining NaNs (e.g., if a whole module group was NaN) with 0 or global median/mean
    remaining_nans = df[columns_to_impute].isnull().sum().sum()
    if remaining_nans > 0:
        print(f"  Warning: {remaining_nans} NaNs remaining after group imputation. Filling with 0.")
        df[columns_to_impute] = df[columns_to_impute].fillna(0) # Simple fill for any leftovers

    imputed_counts = original_nan_counts - df[columns_to_impute].isnull().sum()
    print("  Missing values imputed:")
    print(imputed_counts[imputed_counts > 0]) # Show counts only for columns that had NaNs imputed
    return df

def handle_outliers(df, columns_to_clip, lower_quantile=0.01, upper_quantile=0.99):
    """
    Handles outliers by clipping values to specified quantiles.
    Clipping replaces values outside the quantile range with the quantile boundary value.
    This is done per module if Module_ID exists.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_to_clip (list): List of column names to handle outliers for.
        lower_quantile (float): Lower quantile boundary (e.g., 0.01 for 1st percentile).
        upper_quantile (float): Upper quantile boundary (e.g., 0.99 for 99th percentile).

    Returns:
        pd.DataFrame: DataFrame with outliers clipped.
    """
    print(f"Handling outliers using clipping between quantiles ({lower_quantile}, {upper_quantile})...")

    if 'Module_ID' not in df.columns:
        print("  Warning: 'Module_ID' not found. Clipping globally instead of per module.")
        for col in columns_to_clip:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                low = df[col].quantile(lower_quantile)
                high = df[col].quantile(upper_quantile)
                original_sum = df[col].sum() # For checking change
                df[col] = df[col].clip(lower=low, upper=high)
                if df[col].sum() != original_sum:
                     print(f"  Clipped outliers in column: {col} (Bounds: [{low:.2f}, {high:.2f}])")
    else:
        print("  Clipping outliers per Module_ID group...")
        clipped_cols = []
        for col in columns_to_clip:
             if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Define a function to clip within a group
                def clip_group(group):
                    low = group.quantile(lower_quantile)
                    high = group.quantile(upper_quantile)
                    return group.clip(lower=low, upper=high)

                # Apply the clipping function to each group
                original_sum = df[col].sum()
                df[col] = df.groupby('Module_ID')[col].transform(clip_group)
                if df[col].sum() != original_sum:
                     clipped_cols.append(col)
        if clipped_cols:
            print(f"  Clipped outliers in columns: {clipped_cols}")
        else:
            print("  No outliers needed clipping based on specified quantiles.")

    return df

# --- Feature Engineering Functions ---

def calculate_delta_features(df, columns, window=1, group_col='Module_ID'):
    """
    Calculates the difference (delta) between the current value and the value 'window' steps ago.
    Operates per group (e.g., per Module_ID).

    Args:
        df (pd.DataFrame): Input DataFrame (sorted by group_col and Timestamp).
        columns (list): List of column names to calculate delta for.
        window (int): Number of time steps to look back for the difference.
        group_col (str): Column name to group by (e.g., 'Module_ID').

    Returns:
        pd.DataFrame: DataFrame with new delta columns added (e.g., 'Delta_1_Temp_C').
    """
    print(f"Calculating delta features (window={window})...")
    new_cols = {}
    if group_col not in df.columns:
        print(f"  Warning: Group column '{group_col}' not found. Calculating global deltas.")
        for col in columns:
            if col in df.columns:
                delta_col_name = f"Delta_{window}_{col}"
                new_cols[delta_col_name] = df[col].diff(periods=window)
                print(f"  Created global delta column: {delta_col_name}")
    else:
        print(f"  Calculating deltas per '{group_col}' group...")
        grouped = df.groupby(group_col)
        for col in columns:
             if col in df.columns:
                delta_col_name = f"Delta_{window}_{col}"
                # Use transform with diff to align results correctly with original df index
                new_cols[delta_col_name] = grouped[col].transform(lambda x: x.diff(periods=window))
                print(f"  Created grouped delta column: {delta_col_name}")

    # Add new columns to the DataFrame
    for name, data in new_cols.items():
        df[name] = data

    # Deltas will introduce NaNs at the start of each group/series, fill them
    delta_cols_to_fill = list(new_cols.keys())
    print(f"  Filling initial NaNs in delta columns with 0...")
    df[delta_cols_to_fill] = df[delta_cols_to_fill].fillna(0)

    return df

def calculate_rolling_features(df, columns, windows, stats=['mean', 'std'], group_col='Module_ID'):
    """
    Calculates rolling statistics (mean, std dev) over specified windows.
    Operates per group (e.g., per Module_ID).

    Args:
        df (pd.DataFrame): Input DataFrame (sorted by group_col and Timestamp).
        columns (list): List of column names to calculate rolling stats for.
        windows (list of int): List of window sizes (number of time steps).
        stats (list of str): List of statistics to calculate ('mean', 'std', 'min', 'max', etc.).
        group_col (str): Column name to group by (e.g., 'Module_ID').

    Returns:
        pd.DataFrame: DataFrame with new rolling feature columns added.
    """
    print(f"Calculating rolling features (windows={windows}, stats={stats})...")
    if group_col not in df.columns:
        print(f"  Warning: Group column '{group_col}' not found. Calculating global rolling stats.")
        for col in columns:
            if col in df.columns:
                for window in windows:
                    for stat in stats:
                        new_col_name = f"Rolling_{stat}_{window}_{col}"
                        # Calculate rolling stat
                        rolling_result = df[col].rolling(window=window, min_periods=1).agg(stat)
                        df[new_col_name] = rolling_result
                        print(f"  Created global rolling column: {new_col_name}")
    else:
        print(f"  Calculating rolling stats per '{group_col}' group...")
        grouped = df.groupby(group_col)
        for col in columns:
            if col in df.columns:
                for window in windows:
                    for stat in stats:
                        new_col_name = f"Rolling_{stat}_{window}_{col}"
                        # Use transform with rolling to align results
                        # Note: transform doesn't directly support multiple rolling stats easily.
                        # We calculate per group and assign back. Need to handle index alignment carefully.
                        # Let's calculate directly and assign (requires index to be unique or reset/set later)
                        df[new_col_name] = grouped[col].transform(lambda x: x.rolling(window=window, min_periods=1).agg(stat))
                        print(f"  Created grouped rolling column: {new_col_name}")

    # Rolling features will introduce NaNs at the start, fill them (e.g., with 0 or backfill)
    rolling_cols = [f"Rolling_{stat}_{w}_{col}" for col in columns for w in windows for stat in stats if f"Rolling_{stat}_{w}_{col}" in df.columns]
    print(f"  Filling initial NaNs in rolling columns with 0...")
    df[rolling_cols] = df[rolling_cols].fillna(0) # Or use bfill within groups if more appropriate

    return df


# --- Main Pipeline Function ---

def run_preprocessing_pipeline(raw_data_path, processed_data_path):
    """
    Runs the full preprocessing and feature engineering pipeline.

    Args:
        raw_data_path (str): Path to the raw input CSV file.
        processed_data_path (str): Path where the processed CSV file should be saved.
    """
    # 1. Load Data
    df_raw = load_data(raw_data_path)
    if df_raw.empty:
        return None

    # 2. Handle Timestamp and Sort
    df_processed = handle_timestamp(df_raw)

    # 3. Impute Missing Values
    # Forward fill is often suitable for time-series sensor data, assuming values don't change instantly
    df_processed = impute_missing_values(df_processed, SENSOR_COLUMNS, method='ffill')

    # 4. Handle Outliers (Optional - Apply carefully)
    # Clipping extreme values that might be sensor errors but keeping high values from faults
    # We apply this only to sensor columns, not gas proxies which can legitimately spike high
    columns_to_clip = ['Module_Avg_Surface_Temp_C', 'Module_Max_Surface_Temp_C',
                       'Module_Voltage_V', 'Module_Current_A', 'Module_SoC_percent',
                       'Ambient_Temp_Rack_C']
    df_processed = handle_outliers(df_processed, columns_to_clip, lower_quantile=0.01, upper_quantile=0.99)

    # 5. Feature Engineering
    # a) Calculate delta (rate of change over 1 time step) for key sensors
    delta_cols = ['Module_Avg_Surface_Temp_C', 'Module_Voltage_V', 'Module_SoC_percent']
    df_processed = calculate_delta_features(df_processed, delta_cols, window=1) # Delta over 10 seconds

    # b) Calculate rolling statistics (mean and std dev) over different windows
    rolling_cols = ['Module_Avg_Surface_Temp_C', 'Module_Voltage_V', 'Module_Current_A']
    # Windows corresponding to ~1 min, 5 min, 10 min
    rolling_windows = [6, 30, 60] # 6*10s=1min, 30*10s=5min, 60*10s=10min
    df_processed = calculate_rolling_features(df_processed, rolling_cols, rolling_windows, stats=['mean', 'std'])

    # 6. Select Final Features (Example: Keep all for now, selection happens before training)
    # In a real scenario, you might drop intermediate columns or select based on analysis
    final_columns = df_processed.columns.tolist() # Keep all generated columns for now
    print(f"\nFinal selected columns ({len(final_columns)}): {final_columns}")

    # 7. Save Processed Data
    print(f"\nSaving processed data to: {processed_data_path}")
    # Ensure directory exists
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df_processed.to_csv(processed_data_path, index=False) # Save without index if Timestamp is a column
    print("Preprocessing and feature engineering complete.")

    return df_processed

# --- Example Usage ---
if __name__ == '__main__':
    # This block executes only when the script is run directly
    print("Running preprocessing pipeline as standalone script...")

    # Define paths relative to the project root (assuming this script is in src/preprocessing)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Go up two levels
    
    # *** MAKE SURE THIS FILENAME MATCHES YOUR LATEST DUMMY DATA ***
    RAW_DATA_FILENAME = "dummy_fault_RackA2_Module01_200rows.csv"
    
    INPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'raw', RAW_DATA_FILENAME)
    OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', f"processed_{RAW_DATA_FILENAME}")

    # Run the pipeline
    try:
        df_final = run_preprocessing_pipeline(INPUT_CSV, OUTPUT_CSV)
        if df_final is not None:
            print("\n--- Processed DataFrame Info ---")
            df_final.info()
            print("\n--- Processed DataFrame Head ---")
            print(df_final.head())
            print("\n--- Check NaNs in Processed Data (Should be 0) ---")
            print(df_final.isnull().sum().sum())
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

