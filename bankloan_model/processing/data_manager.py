import os
import shutil
import logging
from pathlib import Path
from typing import List
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from bankloan_model.config.core import config, TRAINED_MODEL_DIR, DATASET_DIR
from bankloan_model import __version__ as _version

logger = logging.getLogger(__name__)

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Loads the bankloan data from a file."""
    dataframe = pd.read_csv(Path(DATASET_DIR)/ file_name)
    dataframe.rename(columns={'cb_person_default_on_file': 'previous_loan_defaults_on_file'}, inplace=True)
    return dataframe

def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    
#    Remove old model pipelines.
#    This is to ensure not too much memory is taken up.
    
#    logger.info(f"Cleaning up old model pipelines in {config.TRAINED_MODEL_DIR}")
#    logger.info(f"Files to keep: {files_to_keep}")
    
    try:
        # Get all items in directory and subdirectories
        for item in Path(TRAINED_MODEL_DIR).glob('*'):
            if item.is_file() and item.suffix == '.pkl' and item.name not in files_to_keep:
                try:
                    logger.info(f"Attempting to remove: {item}")
                    os.remove(item)
                    logger.info(f"Successfully removed: {item}")
                except OSError as e:
                    logger.error(f"Error removing {item}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error during pipeline cleanup: {e}")
        # Continue execution even if cleanup fails
        pass


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """
    Persist the pipeline.
    Saves the versioned model, and overwrites any past models.
    """
    try:
        # Debug current state
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"TRAINED_MODEL_DIR path: {TRAINED_MODEL_DIR}")
        logger.info(f"TRAINED_MODEL_DIR exists: {TRAINED_MODEL_DIR.exists()}")

        # Ensure directory exists
        TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified directory: {TRAINED_MODEL_DIR}")
        
        # Prepare versioned save file name
        save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
        save_path = TRAINED_MODEL_DIR / save_file_name
        logger.info(f"Full save path: {save_path}")

        # Remove old pipelines
        remove_old_pipelines(files_to_keep=[save_file_name])
        
        # Save the pipeline
        logger.info("Attempting to save pipeline...")
        joblib.dump(pipeline_to_persist, save_path)
        logger.info(f"Pipeline successfully saved to: {save_path}")

    except Exception as e:
        logger.error(f"Error saving pipeline: {str(e)}")
        raise
    
def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""
    try:
        file_path = TRAINED_MODEL_DIR / file_name
        logger.info(f"Loading pipeline from: {file_path}")
        trained_model = joblib.load(filename=file_path)
        logger.info("Pipeline loaded successfully")
        return trained_model
    except Exception as e:
        logger.error(f"Error loading the pipeline: {str(e)}")
        raise

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare data before pipeline processing."""
    
    # Make a copy of the dataframe
    data = data_frame.copy()
    
    # Drop unnecessary columns if they exist
    if "loan_status" in data.columns:
        data.drop(labels=["loan_status"], axis=1, inplace=True)
        
    # Rename columns for consistency
    if "cb_person_default_on_file" in data.columns:
        data.rename(
            columns={"cb_person_default_on_file": "previous_loan_defaults_on_file"}, 
            inplace=True
        )
        
    logger.info("Data preparation completed")
    return data