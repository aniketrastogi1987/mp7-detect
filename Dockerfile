# pull python base image
FROM python:3.10-slim

# copy application files
ADD /patient_model_api /patient_model_api/
ADD /patient_model_api/requirements.txt .
ADD /patient_model_api/*.whl .

# specify working directory
WORKDIR /patient_model_api

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# copy application files
ADD /patient_model_api/app/* ./app/


# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]
