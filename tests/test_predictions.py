import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import accuracy_score

from patient_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 60

    # When
    X, y = sample_input_data
    input_data = X.iloc[:expected_no_predictions].copy()
    input_data['platelets'] = input_data['platelets'].astype(float)
    result = make_prediction(input_data=input_data)

    # Then
    assert isinstance(result, dict)
    assert isinstance(result.get('predictions'), list)
    assert isinstance(result.get('version'), str)
    assert len(result.get('predictions')) == expected_no_predictions