import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from patient_model import __version__ as _version
from patient_model.config.core import config
from patient_model.pipeline import patient_pipe
from patient_model.processing.data_manager import load_pipeline
from patient_model.processing.data_manager import pre_pipeline_preparation
from patient_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
patient_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_df=data)
    print(f"Validation errors: {errors}")
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    print(validated_data)
    
    results = {"predictions": None, "version": _version, "errors": errors}
    #predictions = patient_pipe.predict(validated_data)

    try:
        predictions = patient_pipe.predict(validated_data)
        results = {
            "predictions": predictions.tolist(),
            "version": _version,
            "errors": errors
        }
        print(f"Prediction results: {results}")  # Debug output
    except Exception as e:
        results["errors"] = f"Error making prediction: {str(e)}"
        print(f"Prediction error: {str(e)}")  # Debug output
    
    return results
    
if __name__ == "__main__":

    data_in={'age':[48],
             'anaemia':[0],
             'creatinine_phosphokinase':[80],
             'diabetes':[1],
             'ejection_fraction':[65],
             'high_blood_pressure':[0],
             'platelets':[275000],
             'serum_creatinine':[1],
             'serum_sodium':[138],
             'sex':[1],
             'smoking':[1],
             'time':[14], 
             }

    make_prediction(input_data=data_in)
