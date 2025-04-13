import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import accuracy_score

from bankloan_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 179

    # When
    X, y = sample_input_data
    input_data = X.iloc[:expected_no_predictions].to_dict(orient="records")
    result = make_prediction(input_data=input_data)

    # Then
    predictions = result["predictions"]
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == expected_no_predictions