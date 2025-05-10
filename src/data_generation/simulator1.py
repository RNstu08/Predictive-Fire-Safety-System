# src/data_generation/simulator.py

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import uuid
import os

# --- Configuration Constants ---
RACK_IDS_FORMAT = ["RackA{}", "RackB{}"]
NUM_RACKS_PER_ROW = 4
MODULES_PER_RACK = 8
DEFAULT_TIME_STEP_S = 10
SECONDS_IN_HOUR = 3600
SECONDS_IN_DAY = 24 * SECONDS_IN_HOUR
NOMINAL_CELL_VOLTAGE = 3.7
CELL_CAPACITY_AH = 50
MODULE_CELLS_SERIES = 14
MODULE_CELLS_PARALLEL = 2
MODULE_NOMINAL_CAPACITY_AH = CELL_CAPACITY_AH * MODULE_CELLS_PARALLEL
MODULE_NOMINAL_VOLTAGE = NOMINAL_CELL_VOLTAGE * MODULE_CELLS_SERIES
MODULE_NOMINAL_ENERGY_KWH = (MODULE_NOMINAL_VOLTAGE * MODULE_NOMINAL_CAPACITY_AH) / 1000
# Base Thermal Properties
BASE_MODULE_CORE_THERMAL_MASS_J_PER_C = 40000
BASE_MODULE_SURFACE_THERMAL_MASS_J_PER_C = 10000
MODULE_CORE_TO_SURFACE_HT_W_PER_K = 20.0
MODULE_SURFACE_TO_AMBIENT_HT_W_PER_K = 10.0
# Electrical Properties
MODULE_BASE_INTERNAL_RESISTANCE_OHM = 0.005
MODULE_RESISTANCE_TEMP_COEFF = 0.0001
MODULE_RESISTANCE_SOC_LOW_MULTIPLIER = 1.8
MODULE_RESISTANCE_SOC_HIGH_MULTIPLIER = 1.4
# Failure Trigger Conditions (PNR)
PNR_INTERNAL_TEMP_C = 140.0 # Reset to original value
PNR_DELTA_T_INTERNAL_PER_SECOND_C = 0.5
# Off-Gassing Proxies Trigger Temperatures
OFFGAS_CO_TRIGGER_INTERNAL_TEMP_C = 75.0
OFFGAS_H2_TRIGGER_INTERNAL_TEMP_C = 95.0
OFFGAS_VOC_TRIGGER_INTERNAL_TEMP_C = 110.0
OFFGAS_INCREASE_RATE_FACTOR = 0.05
# --- WIDENED Prediction Window for Labeling ---
PREDICTION_WINDOW_SECONDS = 20 * 60  # Increased to 20 minutes
GUARD_BUFFER_SECONDS = 30            # Reduced to 30 seconds
# --- END WIDENED WINDOW ---
# Data Realism Parameters
SENSOR_NOISE_STD_DEV_ABS = {
    'Module_Avg_Surface_Temp_C': 0.2, 'Module_Max_Surface_Temp_C': 0.25,
    'Module_Voltage_V': 0.05, 'Module_Current_A': 0.1,
    'Sim_OffGas_CO_ppm_proxy': 5.0, 'Sim_OffGas_H2_ppm_proxy': 5.0,
    'Sim_OffGas_VOC_ppm_proxy': 5.0, 'Ambient_Temp_Rack_C': 0.1
}
SENSOR_MISSING_RATE_PERCENT = 0.05
SENSOR_OUTLIER_RATE_PERCENT = 0.01
OUTLIER_MAGNITUDE_STD_DEV_FACTOR = 7

# --- Utility Functions ---
# (generate_module_id, add_noise_to_value, introduce_missing_value_randomly, introduce_outlier_randomly remain the same)
def generate_module_id(rack_id_format_str, rack_num_in_row, module_index_in_rack):
    rack_id = rack_id_format_str.format(rack_num_in_row)
    return f"{rack_id}_Module{module_index_in_rack:02d}"

def add_noise_to_value(value, key_for_noise_config):
    if pd.isna(value): return value
    std_dev = SENSOR_NOISE_STD_DEV_ABS.get(key_for_noise_config, 0.01)
    return value + np.random.normal(0, std_dev)

def introduce_missing_value_randomly(value, missing_rate_percent):
    if random.uniform(0, 100) < missing_rate_percent: return np.nan
    return value

def introduce_outlier_randomly(value, key_for_noise_config, outlier_rate_percent):
    if pd.isna(value): return value
    if random.uniform(0, 100) < outlier_rate_percent:
        std_dev = SENSOR_NOISE_STD_DEV_ABS.get(key_for_noise_config, 0.01)
        return value + np.random.choice([-1, 1]) * OUTLIER_MAGNITUDE_STD_DEV_FACTOR * std_dev
    return value

# --- BatteryModule Class ---
class BatteryModule:
    def __init__(self, module_id, rack_id, initial_soc=None, initial_internal_temp_c=25.0, initial_surface_temp_c=24.5):
        self.module_id = module_id
        self.rack_id = rack_id
        self.internal_unique_id = str(uuid.uuid4())
        self.soc = initial_soc if initial_soc is not None else random.uniform(0.5, 0.95)
        self.voltage_v = self._calculate_ocv(self.soc)
        self.current_a = 0.0
        self.internal_resistance_ohm = self._calculate_internal_r(self.soc, initial_internal_temp_c)
        self.internal_cell_temp_c = initial_internal_temp_c
        self.surface_temp_c = initial_surface_temp_c
        self.prev_internal_temp_c = initial_internal_temp_c
        self.delta_internal_temp_per_s = 0.0
        self.health_status = "Normal"
        self.fault_active = False
        self.fault_type = None
        self.fault_params = {}
        self.fault_progression_time_s = 0.0
        self.pnr_triggered_time_s = np.nan
        self.sim_co_ppm = 0.0
        self.sim_h2_ppm = 0.0
        self.sim_voc_ppm = 0.0
        self.base_core_thermal_mass = BASE_MODULE_CORE_THERMAL_MASS_J_PER_C
        self.base_surface_thermal_mass = BASE_MODULE_SURFACE_THERMAL_MASS_J_PER_C

    def _calculate_ocv(self, soc):
        min_cell_ocv, max_cell_ocv = 3.2, 4.2
        ocv_per_cell = min_cell_ocv + (max_cell_ocv - min_cell_ocv) * soc
        return np.clip(ocv_per_cell, 3.0, 4.2) * MODULE_CELLS_SERIES

    def _calculate_internal_r(self, soc, temp_c):
        r = MODULE_BASE_INTERNAL_RESISTANCE_OHM * (1 + MODULE_RESISTANCE_TEMP_COEFF * abs(temp_c - 25.0))
        if soc < 0.1: r *= MODULE_RESISTANCE_SOC_LOW_MULTIPLIER + (0.1 - soc) * 20
        elif soc > 0.9: r *= MODULE_RESISTANCE_SOC_HIGH_MULTIPLIER + (soc - 0.9) * 10
        return max(r, MODULE_BASE_INTERNAL_RESISTANCE_OHM * 0.7)

    def _update_soc(self, time_step_s):
        delta_charge_ah = (self.current_a * time_step_s) / SECONDS_IN_HOUR
        self.soc = np.clip(self.soc + delta_charge_ah / MODULE_NOMINAL_CAPACITY_AH, 0.0, 1.0)

    def _update_electrical_model(self):
        self.internal_resistance_ohm = self._calculate_internal_r(self.soc, self.internal_cell_temp_c)
        self.voltage_v = self._calculate_ocv(self.soc) + (self.current_a * self.internal_resistance_ohm)

    def _update_thermal_model(self, current_sim_time_s, time_step_s, rack_ambient_temp_c, external_heat_input_W=0):
        p_gen_ohmic_W = (self.current_a**2) * self.internal_resistance_ohm
        p_gen_fault_W = 0.0
        core_thermal_mass = self.base_core_thermal_mass
        surface_thermal_mass = self.base_surface_thermal_mass

        if self.fault_active and self.fault_type == 'ISC':
            base_heat_W = float(self.fault_params.get('base_isc_heat_W', 50.0))
            max_heat_W = float(self.fault_params.get('max_isc_heat_W', 5000.0))
            time_to_reach_max_heat_s = float(self.fault_params.get('time_to_max_heat_s', 1800.0))

            if time_to_reach_max_heat_s <= 0:
                 calculated_fault_heat = max_heat_W
            elif self.fault_progression_time_s < time_to_reach_max_heat_s:
                progression_ratio = self.fault_progression_time_s / time_to_reach_max_heat_s
                calculated_fault_heat = base_heat_W + (max_heat_W - base_heat_W) * progression_ratio
            else:
                calculated_fault_heat = max_heat_W
            p_gen_fault_W = min(calculated_fault_heat, max_heat_W)

        total_heat_gen_core_W = p_gen_ohmic_W + p_gen_fault_W
        p_core_to_surface_W = MODULE_CORE_TO_SURFACE_HT_W_PER_K * (self.internal_cell_temp_c - self.surface_temp_c)
        p_surface_to_ambient_W = MODULE_SURFACE_TO_AMBIENT_HT_W_PER_K * (self.surface_temp_c - rack_ambient_temp_c)

        net_heat_into_core_W = total_heat_gen_core_W - p_core_to_surface_W
        delta_T_core_C = (net_heat_into_core_W * time_step_s) / core_thermal_mass if core_thermal_mass > 0 else 0
        self.internal_cell_temp_c += delta_T_core_C

        net_heat_into_surface_W = p_core_to_surface_W - p_surface_to_ambient_W + external_heat_input_W
        delta_T_surface_C = (net_heat_into_surface_W * time_step_s) / surface_thermal_mass if surface_thermal_mass > 0 else 0
        self.surface_temp_c += delta_T_surface_C

        self.delta_internal_temp_per_s = (self.internal_cell_temp_c - self.prev_internal_temp_c) / time_step_s if time_step_s > 0 else 0
        self.prev_internal_temp_c = self.internal_cell_temp_c

        self.internal_cell_temp_c = max(self.internal_cell_temp_c, -50.0)
        self.surface_temp_c = max(self.surface_temp_c, -50.0)

    def _update_failure_progression_and_offgassing(self, current_sim_time_s, time_step_s):
        if self.fault_active:
            self.fault_progression_time_s += time_step_s

            if pd.isna(self.pnr_triggered_time_s):
                pnr_by_temp = self.internal_cell_temp_c >= PNR_INTERNAL_TEMP_C
                pnr_by_rate = (self.delta_internal_temp_per_s >= PNR_DELTA_T_INTERNAL_PER_SECOND_C and
                               self.internal_cell_temp_c > OFFGAS_VOC_TRIGGER_INTERNAL_TEMP_C)

                if pnr_by_temp or pnr_by_rate:
                    self.pnr_triggered_time_s = current_sim_time_s
                    self.health_status = "Runaway"
                    # Keep PNR trigger debug print
                    if self.module_id == "RackA2_Module01":
                        print(f"DEBUGGER: PNR TRIGGERED for {self.module_id} at sim_time {current_sim_time_s:.0f}s (fault_time {self.fault_progression_time_s:.0f}s).\n"
                              f"  Trigger: temp={pnr_by_temp}, rate={pnr_by_rate}.\n"
                              f"  Values: IntTemp={self.internal_cell_temp_c:.2f}°C, DeltaT_rate={self.delta_internal_temp_per_s:.3f}°C/s")

            if not pd.isna(self.pnr_triggered_time_s) and current_sim_time_s >= self.pnr_triggered_time_s:
                 self.health_status = "Runaway"
            elif self.fault_active and self.internal_cell_temp_c > OFFGAS_VOC_TRIGGER_INTERNAL_TEMP_C:
                 self.health_status = "PreRunaway"
            elif self.fault_active:
                 self.health_status = "Degrading"

        if self.internal_cell_temp_c > OFFGAS_CO_TRIGGER_INTERNAL_TEMP_C and self.delta_internal_temp_per_s > 0.001:
            self.sim_co_ppm += OFFGAS_INCREASE_RATE_FACTOR * (self.internal_cell_temp_c - OFFGAS_CO_TRIGGER_INTERNAL_TEMP_C) * time_step_s
        if self.internal_cell_temp_c > OFFGAS_H2_TRIGGER_INTERNAL_TEMP_C and self.delta_internal_temp_per_s > 0.001:
            self.sim_h2_ppm += OFFGAS_INCREASE_RATE_FACTOR * (self.internal_cell_temp_c - OFFGAS_H2_TRIGGER_INTERNAL_TEMP_C) * time_step_s
        if self.internal_cell_temp_c > OFFGAS_VOC_TRIGGER_INTERNAL_TEMP_C and self.delta_internal_temp_per_s > 0.001:
            self.sim_voc_ppm += OFFGAS_INCREASE_RATE_FACTOR * (self.internal_cell_temp_c - OFFGAS_VOC_TRIGGER_INTERNAL_TEMP_C) * time_step_s
        self.sim_co_ppm = max(0, min(self.sim_co_ppm, 2000))
        self.sim_h2_ppm = max(0, min(self.sim_h2_ppm, 2000))
        self.sim_voc_ppm = max(0, min(self.sim_voc_ppm, 2000))

    def introduce_fault(self, fault_type_str, fault_activation_time_s, current_simulation_time_s, **kwargs_fault_params):
        if not self.fault_active and current_simulation_time_s >= fault_activation_time_s:
            self.fault_active = True
            self.fault_type = fault_type_str
            self.fault_params = kwargs_fault_params
            self.fault_progression_time_s = 0.0
            self.health_status = "Degrading"
            if self.module_id == "RackA2_Module01":
                print(f"DEBUGGER: Fault '{self.fault_type}' ACTIVATED for {self.module_id} at sim_time {current_simulation_time_s:.0f}s. Params: {self.fault_params}")

    def update_state(self, current_sim_time_s, time_step_s, rack_ambient_temp_c, assigned_load_current_a, external_heat_input_W=0):
        self.current_a = assigned_load_current_a
        self._update_soc(time_step_s)
        self._update_electrical_model()
        self._update_failure_progression_and_offgassing(current_sim_time_s, time_step_s)
        self._update_thermal_model(current_sim_time_s, time_step_s, rack_ambient_temp_c, external_heat_input_W)

    def get_sensor_readings(self, current_sim_time_s, rack_ambient_temp_c):
        true_readings = {
            'Timestamp': pd.Timestamp(datetime(2024, 1, 1, 0, 0, 0)) + pd.Timedelta(seconds=current_sim_time_s),
            'Module_ID': self.module_id, 'Rack_ID': self.rack_id,
            'Module_Avg_Surface_Temp_C': self.surface_temp_c,
            'Module_Max_Surface_Temp_C': self.surface_temp_c * random.uniform(1.0, 1.015),
            'Module_Voltage_V': self.voltage_v, 'Module_Current_A': self.current_a,
            'Module_SoC_percent': self.soc * 100.0,
            'Sim_OffGas_CO_ppm_proxy': self.sim_co_ppm,
            'Sim_OffGas_H2_ppm_proxy': self.sim_h2_ppm,
            'Sim_OffGas_VOC_ppm_proxy': self.sim_voc_ppm,
            'Ambient_Temp_Rack_C': rack_ambient_temp_c,
        }
        imperfect_readings = {}
        for key, true_value in true_readings.items():
            if key in SENSOR_NOISE_STD_DEV_ABS:
                val_noisy = add_noise_to_value(true_value, key)
                val_missing = introduce_missing_value_randomly(val_noisy, SENSOR_MISSING_RATE_PERCENT)
                final_val = introduce_outlier_randomly(val_missing, key, SENSOR_OUTLIER_RATE_PERCENT)
                imperfect_readings[key] = final_val
            else:
                imperfect_readings[key] = true_value
        return imperfect_readings

    # --- ADDED DEBUG PRINT TO GET_LABEL ---
    def get_label(self, current_sim_time_s):
        if pd.isna(self.pnr_triggered_time_s): return 0
        # Use WIDENED window constants
        start = self.pnr_triggered_time_s - PREDICTION_WINDOW_SECONDS - GUARD_BUFFER_SECONDS
        end = self.pnr_triggered_time_s - GUARD_BUFFER_SECONDS
        is_hazardous = start <= current_sim_time_s < end
        if is_hazardous:
            # Print every time for debugging clarity
            print(f"DEBUGGER Label=1: Module={self.module_id} | Time={current_sim_time_s:.0f}s | PNR at {self.pnr_triggered_time_s:.0f}s | Window [{start:.0f}s, {end:.0f}s)")
            return 1
        else:
            return 0
    # --- END ADDED DEBUG PRINT ---

# --- WarehouseSimulator Class ---
class WarehouseSimulator:
    def __init__(self, rack_id_formats_list, num_racks_per_fmt_row, modules_per_single_rack,
                 initial_global_warehouse_ambient_temp_c=22.0):
        self.modules = []
        self.module_map_by_id = {}
        self.rack_to_module_ids_map = {}
        self.current_sim_time_s = 0.0
        self.global_warehouse_ambient_temp_c = initial_global_warehouse_ambient_temp_c
        self.rack_specific_ambient_temps_c = {}
        self.active_scenario_conditions = []
        module_counter = 0
        for fmt_str in rack_id_formats_list:
            for i in range(1, num_racks_per_fmt_row + 1):
                rack_id = fmt_str.format(i)
                self.rack_to_module_ids_map[rack_id] = []
                self.rack_specific_ambient_temps_c[rack_id] = initial_global_warehouse_ambient_temp_c
                for j in range(1, modules_per_single_rack + 1):
                    module_id = generate_module_id(fmt_str, i, j)
                    initial_module_internal_temp = initial_global_warehouse_ambient_temp_c + random.uniform(-0.5, 0.5)
                    initial_module_surface_temp = initial_module_internal_temp - random.uniform(0.1, 0.3)
                    module = BatteryModule(module_id, rack_id,
                                           initial_internal_temp_c=initial_module_internal_temp,
                                           initial_surface_temp_c=initial_module_surface_temp)
                    self.modules.append(module)
                    self.module_map_by_id[module_id] = module
                    self.rack_to_module_ids_map[rack_id].append(module_id)
                    module_counter += 1
        print(f"WarehouseSimulator initialized with {module_counter} modules across {len(self.rack_to_module_ids_map)} racks.")

    def _get_module_by_id(self, module_id_str):
        return self.module_map_by_id.get(module_id_str)

    def apply_scenario_conditions_list(self, list_of_scenario_configs):
        self.active_scenario_conditions = list_of_scenario_configs

    def _process_active_scenario_conditions_for_step(self, time_step_s):
        if not self.active_scenario_conditions: return
        for config in self.active_scenario_conditions:
            condition_type = config.get('type')
            activation_time = config.get('activation_time_s', config.get('fault_activation_time_s', 0))
            if self.current_sim_time_s >= activation_time:
                if condition_type == 'internal_fault':
                    module_to_fault = self._get_module_by_id(config['module_id'])
                    if module_to_fault:
                        module_to_fault.introduce_fault(
                            fault_type_str=config.get('fault_name', 'ISC'),
                            fault_activation_time_s=activation_time,
                            current_simulation_time_s=self.current_sim_time_s,
                            **config.get('fault_params', {})
                        )
                elif condition_type == 'rack_cooling_failure':
                    rack_id = config['rack_id']
                    if rack_id in self.rack_specific_ambient_temps_c:
                        rate = config.get('temp_increase_rate_C_per_hour', 0) / SECONDS_IN_HOUR
                        max_offset = config.get('max_temp_offset_C', 20.0)
                        current_temp = self.rack_specific_ambient_temps_c[rack_id]
                        target_max = self.global_warehouse_ambient_temp_c + max_offset
                        if current_temp < target_max:
                            new_temp = current_temp + (rate * time_step_s)
                            self.rack_specific_ambient_temps_c[rack_id] = min(new_temp, target_max)

    def _calculate_and_get_cascading_heat_inputs(self):
        heat_input = {mod_id: 0.0 for mod_id in self.module_map_by_id.keys()}
        HEAT_OUT = 200
        for module in self.modules:
            if module.health_status == "Runaway" and module.internal_cell_temp_c > (PNR_INTERNAL_TEMP_C + 10):
                rack_id = module.rack_id
                try: module_num = int(module.module_id.split("_Module")[-1])
                except ValueError: continue
                for neighbor_id in self.rack_to_module_ids_map.get(rack_id, []):
                    if neighbor_id == module.module_id: continue
                    try:
                        neighbor_num = int(neighbor_id.split("_Module")[-1])
                        if abs(neighbor_num - module_num) == 1: heat_input[neighbor_id] += HEAT_OUT
                    except ValueError: continue
        return heat_input

    def run_single_simulation_step(self, time_step_s, ext_load_profile_func, ext_global_ambient_profile_func):
        self.global_warehouse_ambient_temp_c = ext_global_ambient_profile_func(self.current_sim_time_s)
        self._process_active_scenario_conditions_for_step(time_step_s)
        cascading_heat = self._calculate_and_get_cascading_heat_inputs()
        step_data = []
        for module_obj in self.modules:
            load = ext_load_profile_func(self.current_sim_time_s, module_obj.module_id)
            ambient = self.rack_specific_ambient_temps_c.get(module_obj.rack_id, self.global_warehouse_ambient_temp_c)
            ext_heat = cascading_heat.get(module_obj.module_id, 0.0)
            module_obj.update_state(self.current_sim_time_s, time_step_s, ambient, load, ext_heat)
            readings = module_obj.get_sensor_readings(self.current_sim_time_s, ambient)
            # Get label (which now includes a print statement if label is 1)
            label = module_obj.get_label(self.current_sim_time_s)
            readings['Hazard_Label'] = label
            # --- ADDED DEBUG PRINT IN LOOP ---
            if label == 1:
                print(f"DEBUG LOOP CHECK: Label=1 assigned at t={self.current_sim_time_s:.0f}s for {module_obj.module_id}")
            # --- END ADDED DEBUG PRINT ---
            step_data.append(readings)
        self.current_sim_time_s += time_step_s
        return step_data

    def run_full_scenario_simulation(self, total_duration_s, time_step_s, list_of_scenario_configs,
                                     load_profile_func, global_ambient_profile_func, scenario_description=""):
        all_data = []
        self.current_sim_time_s = 0.0
        self.apply_scenario_conditions_list(list_of_scenario_configs)
        num_steps = int(total_duration_s / time_step_s)
        print(f"\nStarting simulation: '{scenario_description}' for {total_duration_s / SECONDS_IN_HOUR:.2f} hours ({num_steps} steps).")
        for step in range(num_steps):
            if step % (max(1, num_steps // 20)) == 0:
                print(f"  Simulating step {step+1}/{num_steps} ({(step/num_steps)*100:.1f}%) "
                      f"(Time: {self.current_sim_time_s / SECONDS_IN_HOUR:.2f}h / {total_duration_s / SECONDS_IN_HOUR:.2f}h)")
            step_data = self.run_single_simulation_step(time_step_s, load_profile_func, global_ambient_profile_func)
            all_data.extend(step_data)
        print(f"Simulation '{scenario_description}' finished.")
        if not all_data: print("Warning: No data generated."); return pd.DataFrame()
        return pd.DataFrame(all_data)

# --- Scenario Definition Helper Functions & Load/Ambient Profiles ---
def typical_daily_ambient_temp_profile(current_sim_time_s, base_temp_c=20.0, daily_fluctuation_c=6.0):
    secs = current_sim_time_s % SECONDS_IN_DAY; frac = secs / SECONDS_IN_DAY
    phase = -0.8 * np.pi; var = (daily_fluctuation_c / 2.0) * np.sin(frac * 2 * np.pi + phase)
    return base_temp_c + var

def standard_charge_discharge_idle_profile(current_sim_time_s, module_id, charge_C=0.5, discharge_C=0.7, charge_h=3, discharge_h=2, idle_h=1):
    cycle_s = (charge_h + idle_h + discharge_h + idle_h) * SECONDS_IN_HOUR
    time_in_cycle = current_sim_time_s % cycle_s
    charge_end = charge_h * SECONDS_IN_HOUR; idle1_end = charge_end + idle_h * SECONDS_IN_HOUR
    discharge_end = idle1_end + discharge_h * SECONDS_IN_HOUR
    charge_A = charge_C * MODULE_NOMINAL_CAPACITY_AH; discharge_A = -abs(discharge_C * MODULE_NOMINAL_CAPACITY_AH)
    try: offset = sum(ord(c) for c in module_id) % int(0.25 * SECONDS_IN_HOUR); time_in_cycle = (current_sim_time_s + offset) % cycle_s
    except TypeError: pass
    if 0 <= time_in_cycle < charge_end: return charge_A * random.uniform(0.95, 1.05)
    elif charge_end <= time_in_cycle < idle1_end: return 0.0
    elif idle1_end <= time_in_cycle < discharge_end: return discharge_A * random.uniform(0.95, 1.05)
    else: return 0.0

# --- Main Execution Block ---
def create_and_save_scenario_data(scenario_label, total_sim_duration_s, scenario_condition_list, data_output_folder="data/raw", run_identifier="run01"):
    print(f"\n--- Generating Data for Scenario: {scenario_label} (Run: {run_identifier}) ---")
    os.makedirs(data_output_folder, exist_ok=True)
    warehouse_sim = WarehouseSimulator(RACK_IDS_FORMAT, NUM_RACKS_PER_ROW, MODULES_PER_RACK, 20.0)
    sim_df = warehouse_sim.run_full_scenario_simulation(total_sim_duration_s, DEFAULT_TIME_STEP_S, scenario_condition_list, standard_charge_discharge_idle_profile, typical_daily_ambient_temp_profile, scenario_label)
    output_filename = "" # Initialize filename
    if not sim_df.empty:
        output_filename = f"{scenario_label.replace(' ', '_').lower()}_{run_identifier}.csv"
        full_output_path = os.path.join(data_output_folder, output_filename)
        sim_df.to_csv(full_output_path, index=False)
        print(f"Scenario '{scenario_label}' data successfully saved to: {full_output_path}")
    else: print(f"Warning: No data generated for scenario '{scenario_label}'.")
    return sim_df, output_filename # Return dataframe and filename

if __name__ == '__main__':
    # Scenario 4: Using the aggressive parameters suggested by user text
    faulty_module_id_s4 = generate_module_id(RACK_IDS_FORMAT[0], 2, 1) # RackA2_Module01
    scenario_4_conditions = [
        {
            'type': 'internal_fault', 'module_id': faulty_module_id_s4,
            'fault_activation_time_s': 1 * SECONDS_IN_HOUR,
            'fault_name': 'ISC', # Use standard name 'ISC'
            'fault_params': {
                'base_isc_heat_W': 300.0,
                'time_to_max_heat_s': 0.25 * SECONDS_IN_HOUR, # 15 minutes = 900 seconds
                'max_isc_heat_W': 9000.0
            }
        }
    ]
    # Generate the data
    generated_df, generated_filename = create_and_save_scenario_data(
        scenario_label=f"InternalFault_AggressiveParamsV2_{faulty_module_id_s4}", # New Label
        total_sim_duration_s=6 * SECONDS_IN_HOUR,
        scenario_condition_list=scenario_4_conditions,
        run_identifier="isc_aggressive_v2_run1" # New Run ID
    )

    # --- ADDED: Quick check after generation ---
    if not generated_df.empty:
        print("\n--- Quick Check of Generated Data ---")
        print("Hazard_Label value counts:")
        print(generated_df['Hazard_Label'].value_counts())
        print("\nFirst few rows with Hazard_Label == 1:")
        hazardous_rows = generated_df[generated_df['Hazard_Label'] == 1]
        if not hazardous_rows.empty:
            # Display more info for hazardous rows
            print(hazardous_rows[['Timestamp', 'Module_ID', 'Hazard_Label', 'Module_Avg_Surface_Temp_C']].head())
        else:
            print("No rows with Hazard_Label == 1 found in the generated DataFrame.")
    else:
        print("\nQuick check skipped as DataFrame is empty.")
    # --- END ADDED CHECK ---

    print("\nScenario generation calls complete.")
    print(f"Generated data saved in '{os.path.abspath('data/raw')}' directory.")

