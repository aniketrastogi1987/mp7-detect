import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

try:
    import gradio as gr
except ImportError as e:
    print(f"Error importing gradio: {e}")
    print("Try: pip install 'gradio>=4.0.0' 'urllib3<2.0.0'")
    sys.exit(1)

#import gradio
from fastapi import FastAPI, Request, Response

import random
import numpy as np
import pandas as pd
from patient_model.processing.data_manager import load_dataset, load_pipeline
from patient_model import __version__ as _version
from patient_model.config.core import config
from sklearn.model_selection import train_test_split
from patient_model.predict import make_prediction
import prometheus_client as prom

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# FastAPI object
app = FastAPI()


acc_metric = prom.Gauge('patient_accuracy_score', 'Accuracy score for few random 100 test samples')
f1_metric = prom.Gauge('patient_f1_score', 'F1 score for few random 100 test samples')
precision_metric = prom.Gauge('patient_precision_score', 'Precision score for few random 100 test samples')
recall_metric = prom.Gauge('patient_recall_score', 'Recall score for few random 100 test samples')

# LOAD TEST DATA
pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
patient_pipe= load_pipeline(file_name=pipeline_file_name)
data = load_dataset(file_name=config.app_config_.training_data_file)    # read complete data

X_train, X_test, y_train, y_test = train_test_split(                   # divide into train and test set
    data[config.model_config_.features],
    data[config.model_config_.target],
    test_size=config.model_config_.test_size,
    random_state=config.model_config_.random_state,
)
test_data = X_test.copy()
test_data['target'] = y_test.values


# Function for updating metrics
def update_metrics():
    global test_data
    max_size = len(test_data)
    
    # Adjust sample size to the available data
    size = min(5, max_size)  # Use 5 or the maximum available rows, whichever is smaller
    
    # Sample rows randomly
    test = test_data.sample(size, random_state=random.randint(0, 50))
    
    # Prediction
    y_pred = patient_pipe.predict(test.iloc[:, :-1])
    
    # Calculate metrics
    acc = round(accuracy_score(test['target'], y_pred), 3)  # Accuracy score
    f1 = round(f1_score(test['target'], y_pred), 3)  # F1 score
    precision = round(precision_score(test['target'], y_pred), 3)  # Precision score
    recall = round(recall_score(test['target'], y_pred), 3)  # Recall score
    
    # Update Prometheus metrics
    acc_metric.set(acc)
    f1_metric.set(f1)
    precision_metric.set(precision)
    recall_metric.set(recall)


@app.get("/metrics")

async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())

################################# Prometheus related code END ######################################################


# UI - Input components
in_age = gr.Textbox(lines=1, placeholder=None, value="48", label='Age')
in_anaemia = gr.Textbox(lines=1, placeholder=None, value="0", label='Anaemia')
in_creatinine_phosphokinase = gr.Textbox(lines=1, placeholder=None, value="80", label='Creatinine Phosphokinase')
in_diabetes = gr.Textbox(lines=1, placeholder=None, value="1", label='Diabetes')
in_ejection_fraction = gr.Textbox(lines=1, placeholder=None, value="65", label='Ejection Fraction')
in_high_blood_pressure = gr.Textbox(lines=1, placeholder=None, value="0", label='High Blood Pressure')
in_platelets = gr.Textbox(lines=1, placeholder=None, value="275000", label='Platelets')
in_serum_creatinine = gr.Textbox(lines=1, placeholder=None, value="1", label='Serum Creatinine')
in_serum_sodium = gr.Textbox(lines=1, placeholder=None, value="138", label='Serum Sodium')
in_sex = gr.Textbox(lines=1, placeholder=None, value="1", label='Sex')
in_smoking = gr.Textbox(lines=1, placeholder=None, value="1", label='Smoking')
in_time = gr.Textbox(lines=1, placeholder=None, value="14", label='Time')


# UI - Output component
out_label = gr.Textbox(type="text", label='Prediction', elem_id="out_textbox")

# Label prediction function
def get_output_label(in_age, in_anaemia, in_creatinine_phosphokinase, in_diabetes, in_ejection_fraction, in_high_blood_pressure, in_platelets, in_serum_creatinine, in_serum_sodium, in_sex, in_smoking, in_time):
    
    input_df = pd.DataFrame({"age": [in_age], 
                             "anaemia": [in_anaemia], 
                             "creatinine_phosphokinase": [in_creatinine_phosphokinase],
                             "diabetes": [in_diabetes], 
                             "ejection_fraction": [in_ejection_fraction], 
                             "high_blood_pressure": [in_high_blood_pressure],
                             "platelets": [float(in_platelets)],
                             "serum_creatinine": [float(in_serum_creatinine)], 
                             "serum_sodium": [in_serum_sodium], 
                             "sex": [in_sex],
                             "smoking": [in_smoking], 
                             "time": [in_time]})
    
    result = make_prediction(input_data=input_df.replace({np.nan: None}))["predictions"]
    label = "Alive" if result[0]==1 else "Dead"
    return label


# Create Gradio interface object
iface = gr.Interface(fn = get_output_label,
                         inputs = [in_age, in_anaemia, in_creatinine_phosphokinase, in_diabetes, in_ejection_fraction, in_high_blood_pressure, in_platelets, in_serum_creatinine, in_serum_sodium, in_sex, in_smoking, in_time],
                         outputs = [out_label],
                         title="Patient heart disease Prediction API",
                         description="Predictive model that answers the question: “Patient conditions that can relate to death due to heart diseases”",
                         allow_flagging='never'
                         )

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gr.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 
