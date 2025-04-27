import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from patient_model.config.core import config
from patient_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    errors = None
    try:
        # Convert DataFrame to list of dictionaries
        input_list = input_df.to_dict(orient="records")
        
        # Validate data using Pydantic schema
        validated_data = [PatientDataInputSchema(**item) for item in input_list]
        validated_data = pd.DataFrame([item.dict() for item in validated_data])
    except ValidationError as e:
        errors = [error.dict() for error in e.errors()]
        validated_data = input_df  # Return original DataFrame in case of error
    
    return validated_data, errors


class PatientDataInputSchema(BaseModel):
    age: Optional[int]
    anaemia: Optional[int]
    creatinine_phosphokinase: Optional[int]
    diabetes: Optional[int]
    ejection_fraction: Optional[int]
    high_blood_pressure: Optional[int]
    platelets: Optional[float]
    serum_creatinine: Optional[float]
    serum_sodium: Optional[int]  # Ensure this is present and of correct type
    sex: Optional[int]
    smoking: Optional[int]
    time: Optional[int]

def validate_inputs(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    
    errors = None
    try:
        # Convert DataFrame to list of dictionaries
        input_list = input_df.to_dict(orient="records")
        
        # Validate data using Pydantic schema
        validated_data = [PatientDataInputSchema(**item) for item in input_list]
        validated_data = pd.DataFrame([item.model_dump() for item in validated_data])  # Use model_dump()
    except ValidationError as e:
        errors = [error.dict() for error in e.errors()]
        validated_data = input_df  # Return original DataFrame in case of error
    
    return validated_data, errors

class MultipleDataInputs(BaseModel):
    inputs: List[PatientDataInputSchema]
