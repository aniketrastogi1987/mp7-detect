import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from bankloan_model.config.core import config
from bankloan_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    errors = None
    try:
        # Convert DataFrame to list of dictionaries
        input_list = input_df.to_dict(orient="records")
        
        # Validate data using Pydantic schema
        validated_data = [BankLoanDataInputSchema(**item) for item in input_list]
        validated_data = pd.DataFrame([item.dict() for item in validated_data])
    except ValidationError as e:
        errors = [error.dict() for error in e.errors()]
        validated_data = input_df  # Return original DataFrame in case of error
    
    return validated_data, errors


class BankLoanDataInputSchema(BaseModel):
    person_age: Optional[int]
    person_gender: Optional[str]
    person_education: Optional[str]
    person_income: Optional[int]
    person_emp_exp: Optional[int]
    person_home_ownership: Optional[str]
    loan_amnt: Optional[int]
    loan_intent: Optional[str]
    loan_int_rate: Optional[float]
    loan_percent_income: Optional[float]  # Ensure this is present and of correct type
    cb_person_cred_hist_length: Optional[int]
    credit_score: Optional[int]
    previous_loan_defaults_on_file: Optional[str]

def validate_inputs(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    
    errors = None
    try:
        # Convert DataFrame to list of dictionaries
        input_list = input_df.to_dict(orient="records")
        
        # Validate data using Pydantic schema
        validated_data = [BankLoanDataInputSchema(**item) for item in input_list]
        validated_data = pd.DataFrame([item.model_dump() for item in validated_data])  # Use model_dump()
    except ValidationError as e:
        errors = [error.dict() for error in e.errors()]
        validated_data = input_df  # Return original DataFrame in case of error
    
    return validated_data, errors

class MultipleDataInputs(BaseModel):
    inputs: List[BankLoanDataInputSchema]
