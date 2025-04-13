from typing import Any, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]

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


class MultipleDataInputs(BaseModel):
    inputs: List[BankLoanDataInputSchema]

