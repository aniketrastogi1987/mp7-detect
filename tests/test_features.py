import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from patient_model.config.core import config
from patient_model.processing.features import OutlierHandler
import pytest


def test_intent_transformer(sample_input_data):
    # Given
    handler = OutlierHandler(
        columns=['creatinine_phosphokinase']
    )
    
    X = sample_input_data[0].copy()
    X = X.reset_index(drop=True)  # Reset the index
    
    # Check if index 60 exists
    if len(X) > 60:
        #assert np.isnan(X.loc[60,'loan_intent'])
        assert X.loc[60, 'creatinine_phosphokinase'] is not None

        # When
        subject = handler.transform(X)

        # Then
        assert subject.loc[60,'creatinine_phosphokinase'] == handler.mappings[X.loc[60,'creatinine_phosphokinase']]
    else:
        pytest.skip("Index 60 does not exist in the DataFrame")