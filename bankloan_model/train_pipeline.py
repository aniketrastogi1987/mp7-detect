import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna
import logging

from bankloan_model.config.core import config, TRAINED_MODEL_DIR
from bankloan_model.pipeline import create_pipeline
from bankloan_model.processing.data_manager import load_dataset, save_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial, X_train, y_train, X_test, y_test):
    """Objective function for Optuna optimization."""
    pipeline = create_pipeline(trial)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)

def run_training() -> None:
    """Train the model."""
    try:
        # Create trained_models directory if it doesn't exist
        TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load data
        data = load_dataset(file_name=config.app_config_.training_data_file)

        # Divide train and test
        X_train, X_test, y_train, y_test = train_test_split(
            data[config.model_config_.features],
            data[config.model_config_.loan_status],
            test_size=config.model_config_.test_size,
            random_state=config.model_config_.random_state,
        )

        # Optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_test, y_test), 
            n_trials=10
        )

        # Get best trial
        best_trial = study.best_trial
        logger.info(f"Best trial: {best_trial.params}")

        # Train final model with best parameters
        final_pipeline = create_pipeline(best_trial)
        final_pipeline.fit(X_train, y_train)

        # Calculate and log performance
        y_pred = final_pipeline.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        logger.info(f"Final model ROC AUC: {auc_score:.4f}")

        # Save the pipeline
        save_pipeline(pipeline_to_persist=final_pipeline)
        logger.info("Pipeline training completed.")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    run_training()