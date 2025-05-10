import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os # Added for saving

# --- Simulation parameters ---
num_modules = 64
simulation_hours = 6
time_step_seconds = 10
steps_per_module = int((simulation_hours * 3600) / time_step_seconds)
total_rows = num_modules * steps_per_module
print(f"Targeting {total_rows} total rows ({num_modules} modules * {steps_per_module} steps/module).")

# --- Data Realism Parameters (Copied from simulator.py) ---
SENSOR_NOISE_STD_DEV_ABS = { # Absolute standard deviation for noise
    'Module_Avg_Surface_Temp_C': 0.2, # °C
    'Module_Max_Surface_Temp_C': 0.25, # °C
    'Module_Voltage_V': 0.05, # Volts
    'Module_Current_A': 0.1, # Amps
    'Sim_OffGas_CO_ppm_proxy': 5.0, # ppm
    'Sim_OffGas_H2_ppm_proxy': 5.0, # ppm
    'Sim_OffGas_VOC_ppm_proxy': 5.0, # ppm
    'Ambient_Temp_Rack_C': 0.1 # °C
}
SENSOR_MISSING_RATE_PERCENT = 0.05 # 0.05% chance of a sensor reading being NaN for a given step
SENSOR_OUTLIER_RATE_PERCENT = 0.01 # 0.01% chance of a spurious outlier reading
OUTLIER_MAGNITUDE_STD_DEV_FACTOR = 7 # Outlier is X standard deviations from the true value
# --- End Data Realism Parameters ---

# --- Faulty Module Configuration ---
faulty_module = "RackA2_Module01"
# Define when the hazardous window starts (index based on steps per module)
# Start after 1 hour (3600s) = index 360
hazard_label_start_idx = int(1.0 * 3600 / time_step_seconds)
# Define the number of hazardous steps (rows) per faulty module
num_hazardous_steps = 200 # Target 200 rows as requested
hazard_label_end_idx = hazard_label_start_idx + num_hazardous_steps
print(f"Faulty Module: {faulty_module}")
print(f"Hazardous window (indices): [{hazard_label_start_idx}, {hazard_label_end_idx})")
print(f"Hazardous window (time): [{hazard_label_start_idx*time_step_seconds/3600:.2f}h, {hazard_label_end_idx*time_step_seconds/3600:.2f}h)")

# Generate module IDs
module_ids = []
rack_ids_format = ["RackA{}", "RackB{}"]
num_racks_per_row = 4
modules_per_rack = 8
for fmt_str in rack_ids_format:
    for i in range(1, num_racks_per_row + 1):
        for j in range(1, modules_per_rack + 1):
             module_id = f"{fmt_str.format(i)}_Module{j:02d}"
             module_ids.append(module_id)

# Generate timestamps for one module's simulation duration
timestamps = [datetime(2024, 1, 1) + timedelta(seconds=i * time_step_seconds) for i in range(steps_per_module)]

# Create base DataFrame row by row
data = []
print("Generating data...")
for module in module_ids:
    # Add some slight variation to the base values per module
    base_temp = 25 + random.uniform(-2, 2)
    base_volt = 51 + random.uniform(-1, 1)
    base_soc = 80 + random.uniform(-5, 5)
    base_co = 2 + random.uniform(-0.5, 0.5)
    base_h2 = 2 + random.uniform(-0.5, 0.5)
    base_voc = 2 + random.uniform(-0.5, 0.5)
    base_amb = 22 + random.uniform(-1, 1)

    for i, ts in enumerate(timestamps):
        # Base normal values with noise
        row = {
            'Timestamp': ts,
            'Module_ID': module,
            'Rack_ID': module.split("_")[0], # Extract Rack ID
            'Module_Avg_Surface_Temp_C': round(base_temp + np.random.normal(0, 0.3) + (i/steps_per_module)*2, 3), # Slight drift over time
            'Module_Max_Surface_Temp_C': round(base_temp + 1 + np.random.normal(0, 0.4) + (i/steps_per_module)*2, 3),
            'Module_Voltage_V': round(base_volt + np.random.normal(0, 0.15) - (i/steps_per_module)*1.5, 3), # Slight voltage drop over time
            'Module_Current_A': round(np.random.normal(0, 0.05), 3), # Mostly idle current noise
            'Module_SoC_percent': round(base_soc + np.random.normal(0, 1.5) - (i/steps_per_module)*5, 3), # Slight SoC drop over time
            'Sim_OffGas_CO_ppm_proxy': max(0, round(base_co + np.random.normal(0, 0.5), 3)), # Ensure non-negative
            'Sim_OffGas_H2_ppm_proxy': max(0, round(base_h2 + np.random.normal(0, 0.5), 3)),
            'Sim_OffGas_VOC_ppm_proxy': max(0, round(base_voc + np.random.normal(0, 0.5), 3)),
            'Ambient_Temp_Rack_C': round(base_amb + np.random.normal(0, 0.1), 3),
            'Hazard_Label': 0 # Default to normal
        }

        # Inject hazard data for the specific faulty module within the defined window
        if module == faulty_module and hazard_label_start_idx <= i < hazard_label_end_idx:
            # Calculate progression within the hazardous window (0 to 1)
            progression = (i - hazard_label_start_idx) / num_hazardous_steps

            # Add significant anomalies that increase during the window
            row['Module_Avg_Surface_Temp_C'] += (10 + 60 * progression**2 + np.random.normal(0, 1)) # Temp increases quadratically
            row['Module_Max_Surface_Temp_C'] = row['Module_Avg_Surface_Temp_C'] + random.uniform(1, 5) # Max is higher than avg
            row['Module_Voltage_V'] -= (2 + 15 * progression + np.random.normal(0, 0.5)) # Voltage drops
            row['Module_SoC_percent'] -= (5 + 40 * progression + np.random.normal(0, 2)) # SoC drops faster
            row['Sim_OffGas_CO_ppm_proxy'] += (50 + 400 * progression + np.random.normal(0, 10)) # Gases increase
            row['Sim_OffGas_H2_ppm_proxy'] += (60 + 500 * progression + np.random.normal(0, 10))
            row['Sim_OffGas_VOC_ppm_proxy'] += (70 + 600 * progression + np.random.normal(0, 10))
            row['Hazard_Label'] = 1 # Set the label to hazardous

            # Ensure values stay within somewhat plausible ranges even during fault (optional clamping)
            row['Module_Voltage_V'] = max(row['Module_Voltage_V'], 30)
            row['Module_SoC_percent'] = max(row['Module_SoC_percent'], 0)

        # --- Add Data Imperfections (Noise is already included, let's add missing values/outliers) ---
        for key in ['Module_Avg_Surface_Temp_C', 'Module_Max_Surface_Temp_C', 'Module_Voltage_V',
                    'Module_Current_A', 'Module_SoC_percent', 'Sim_OffGas_CO_ppm_proxy',
                    'Sim_OffGas_H2_ppm_proxy', 'Sim_OffGas_VOC_ppm_proxy', 'Ambient_Temp_Rack_C']:
             # Check if key exists before attempting to modify (important if columns change)
             if key in row:
                 # Missing values
                 if random.uniform(0, 100) < SENSOR_MISSING_RATE_PERCENT:
                     row[key] = np.nan
                     continue # Skip outlier if missing
                 # Outliers (only apply if value is not NaN)
                 if not pd.isna(row[key]) and random.uniform(0, 100) < SENSOR_OUTLIER_RATE_PERCENT:
                     std_dev = SENSOR_NOISE_STD_DEV_ABS.get(key, 0.1) # Use configured or default std dev
                     row[key] += np.random.choice([-1, 1]) * OUTLIER_MAGNITUDE_STD_DEV_FACTOR * std_dev

        data.append(row)

# Convert list of dictionaries to DataFrame
df_dummy = pd.DataFrame(data)
print("Dummy data generation complete.")
print(f"Final DataFrame shape: {df_dummy.shape}")

# --- Verify Label Counts ---
print("\n--- Quick Check of Generated Data ---")
print("Hazard_Label value counts:")
label_counts = df_dummy['Hazard_Label'].value_counts()
print(label_counts)

print("\nFirst few rows with Hazard_Label == 1:")
hazardous_rows = df_dummy[df_dummy['Hazard_Label'] == 1]
if not hazardous_rows.empty:
    # Display more info for hazardous rows
    print(hazardous_rows[['Timestamp', 'Module_ID', 'Hazard_Label', 'Module_Avg_Surface_Temp_C']].head())
    # Verify the number of hazardous rows for the specific module
    faulty_module_hazardous_count = hazardous_rows[hazardous_rows['Module_ID'] == faulty_module].shape[0]
    print(f"\nNumber of hazardous rows for module {faulty_module}: {faulty_module_hazardous_count}")
    if faulty_module_hazardous_count == num_hazardous_steps:
        print("Correct number of hazardous rows generated for the faulty module.")
    else:
        print(f"WARNING: Expected {num_hazardous_steps} hazardous rows for {faulty_module}, but found {faulty_module_hazardous_count}.")

else:
    print("No rows with Hazard_Label == 1 found in the generated DataFrame.")


# --- Save the Dummy Data ---
DUMMY_DATA_FILENAME = f"dummy_fault_{faulty_module}_{num_hazardous_steps}rows.csv"
DUMMY_DATA_OUTPUT_FOLDER = "data/raw"
os.makedirs(DUMMY_DATA_OUTPUT_FOLDER, exist_ok=True) # Ensure directory exists
# Construct path relative to the project root, assuming this script is run from the root
# If running from src/data_generation, the path needs adjustment (e.g., os.path.join("..", "..", DUMMY_DATA_OUTPUT_FOLDER, DUMMY_DATA_FILENAME))
# Assuming running from project root:
full_output_path = os.path.join(DUMMY_DATA_OUTPUT_FOLDER, DUMMY_DATA_FILENAME)

df_dummy.to_csv(full_output_path, index=False)
print(f"\nDummy data successfully saved to: {full_output_path}")
print(f"Absolute path: {os.path.abspath(full_output_path)}")