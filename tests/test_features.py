import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bankloan_model.config.core import config
from bankloan_model.processing.features import Mapper
import pytest


def test_intent_transformer(sample_input_data):
    # Given
    mapper = Mapper(
        variables=config.model_config_.intent_var, mappings=config.model_config_.intent_mappings # intent
    )
    
    X = sample_input_data[0].copy()
    X = X.reset_index(drop=True)  # Reset the index
    
    # Check if index 17 exists
    if len(X) > 17:
        #assert np.isnan(X.loc[17,'loan_intent'])
        assert X.loc[17, 'loan_intent'] is not None

        # When
        subject = mapper.transform(X)

        # Then
        assert subject.loc[17,'loan_intent'] == mapper.mappings[X.loc[17,'loan_intent']]
    else:
        pytest.skip("Index 17 does not exist in the DataFrame")