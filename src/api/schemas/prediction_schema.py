# src/api/schemas/prediction_schema.py

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any

# Defines the structure expected for a single module's sensor readings
# We make fields optional initially and validate later if needed,
# or require them directly using BaseModel defaults.
# For a robust API, defining expected types is crucial.
class ModuleFeatures(BaseModel):
    # Identifiers (optional in features, but useful for context)
    Module_ID: str | None = None 
    
    # Sensor Readings and Engineered Features - Use names from your PROCESSED data
    # Make sure these match the columns used during training (after preprocessing/feature engineering)
    # Example features (replace/add based on your actual features):
    Module_Avg_Surface_Temp_C: float
    Module_Max_Surface_Temp_C: float
    Module_Voltage_V: float
    Module_Current_A: float
    Module_SoC_percent: float
    Sim_OffGas_CO_ppm_proxy: float
    Sim_OffGas_H2_ppm_proxy: float
    Sim_OffGas_VOC_ppm_proxy: float
    Ambient_Temp_Rack_C: float
    Delta_1_Module_Avg_Surface_Temp_C: float # Example engineered feature
    Delta_1_Module_Voltage_V: float         # Example engineered feature
    Delta_1_Module_SoC_percent: float       # Example engineered feature
    Rolling_mean_6_Module_Avg_Surface_Temp_C: float # Example engineered feature
    Rolling_std_6_Module_Avg_Surface_Temp_C: float  # Example engineered feature
    Rolling_mean_30_Module_Avg_Surface_Temp_C: float
    Rolling_std_30_Module_Avg_Surface_Temp_C: float
    Rolling_mean_60_Module_Avg_Surface_Temp_C: float
    Rolling_std_60_Module_Avg_Surface_Temp_C: float
    Rolling_mean_6_Module_Voltage_V: float
    Rolling_std_6_Module_Voltage_V: float
    Rolling_mean_30_Module_Voltage_V: float
    Rolling_std_30_Module_Voltage_V: float
    Rolling_mean_60_Module_Voltage_V: float
    Rolling_std_60_Module_Voltage_V: float
    Rolling_mean_6_Module_Current_A: float
    Rolling_std_6_Module_Current_A: float
    Rolling_mean_30_Module_Current_A: float
    Rolling_std_30_Module_Current_A: float
    Rolling_mean_60_Module_Current_A: float
    Rolling_std_60_Module_Current_A: float
    
    # Add ALL other features your final model was trained on
    # ... (add more features as needed) ...

    # Example of a validator (optional)
    @validator('Module_SoC_percent')
    def soc_must_be_in_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('SoC must be between 0 and 100')
        return v

# Defines the structure of the input request body (a list of module features)
class PredictionInput(BaseModel):
    readings: List[ModuleFeatures]

# Defines the structure for a single prediction result
class PredictionResult(BaseModel):
    Module_ID: str | None = "Unknown" # Include Module_ID if available in input
    hazard_prediction: int = Field(..., description="Prediction result (0: Normal, 1: Hazardous)")
    hazard_probability: float | None = Field(None, description="Predicted probability of hazard (if available)") # Optional probability

# Defines the structure of the response body (a list of prediction results)
class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    model_version: str | None = None # Optional: Include version of the model used
    error: str | None = None # Optional: Include error message if prediction failed

