import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from bankloan_model import __version__ as model_version
from bankloan_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()



example_input = {
    "inputs": [
        {
            "person_age":40,
            "person_gender":"male",
            "person_education":"Doctorate",
            "person_income":20000,
            "person_emp_exp":4,
            "person_home_ownership":"RENT",
            "loan_amnt":10000,
            "loan_intent":'PERSONAL',
            "loan_int_rate":14.2,
            "loan_percent_income":0.5,
            "cb_person_cred_hist_length":2,
            "credit_score":605, 
            "previous_loan_defaults_on_file":'No'
        }
    ]          
}


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Survival predictions with the bankloan_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results

