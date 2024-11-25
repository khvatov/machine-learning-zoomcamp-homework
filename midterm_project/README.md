## Alex Khvatov Midterm project for ML ZoomCamp class (2024)

### Description

Heart Disease is the leading cause of death globally. In this project I am going to focus on prediction of cardio vascular disease based on patients' data collected. The following project uses data collected in Cleveland Hospital (and several others) and donated in 1988. Since the dataset collected at Cleveland hosplital is the most complete, we are going to foucs on just that dataset in order to produce the best ML model.
By creating an accurate ML model we are going to be able produce a probability of presence of cardio-vascular disease in a patient.


### Project setup:

The main file is [midterm.ipynb](midterm.ipynb). It contains an extensive EDA and feature importance analysis.

In order to run this Jupyter notebook one must have Python 3.11 installed and `pipenv`. 
Once inside this cloned repo directory, execute `pipenv install`; this will install all the necessary dependecies to launch Jupyter and interact with this notebook. You may execute `pipenv shell` in order to swith context to the virtual environment created by pipenv. Execute `jupyter lab` in order to launch Jupyter.


#### Cloud or local deployment
In order to build a Docker image you should be on a computer with _x86_ architecture (not on Mac or other arm-based processor-based machines) because the base layer has only been built for _x86_ machines. You can run `docker build -t zoomcamp-midterm-ak .` in order to build Docker image locally. Then you can create and run Docker container by executing `docker run -p 9696:9696 zoomcamp-midterm-ak`

The [Dockerfile](Dockerfile) included used to build the Docker image uploaded to [Docker hub](https://hub.docker.com/repository/docker/khvatov/zoomcamp-midterm-ak).

You may just pull the docker image or recreate it yourself by running the following command `docker build -t khvatov/zoomcamp-midterm-ak:1.0.0 .` given you are in `midterm` directory, have installed the required libraries via `pipenv install` and executed `python train.py` in order to retrain and save the model. I have included the `model.bin` just in case.

You may run this Docker image using docker run -p 9696:9696 khvatov/zoomcamp-midterm-ak:1.0.0

The webservice is going to be available on the port 9696. Please use [web_client.ipynb](web_client.ipynb) notebook to interact with the webservice.

