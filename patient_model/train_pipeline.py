import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import logging

from patient_model.config.core import config, TRAINED_MODEL_DIR
from patient_model.pipeline import patient_pipe
from patient_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config_.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config_.features],  # predictors
        data[config.model_config_.target],
        test_size=config.model_config_.test_size,
        random_state=config.model_config_.random_state,
    )

    # Pipeline fitting
    #pipeline = patient_pipe()
    patient_pipe.fit(X_train,y_train)

    
    y_pred = patient_pipe.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logging.info(f"Model metrics - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    

    # persist trained model
    save_pipeline(pipeline_to_persist= patient_pipe)
    
if __name__ == "__main__":
    run_training()
