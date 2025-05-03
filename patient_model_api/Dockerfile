# pull python base image
FROM python:3.10-slim

# copy application files
ADD requirements.txt requirements.txt
ADD *.whl .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# copy application files
ADD app/* . /app/


# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]
