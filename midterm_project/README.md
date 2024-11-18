## Alex Khvatov Midterm project for ML ZoomCamp class (2024)


### Project setup:

The main file is [midterm.ipynb](midterm.ipynb).
In order to run this Jupyter notebook one must have Python 3.11 installed and `pipenv`. 
Once inside this cloned repo directory, execute `pipenv install`; this will install all the necessary dependecies to launch Jupyter and interact with this notebook. You may execute `pipenv shell` in order to swith context to the virtual environment created by pipenv. Execute `jupyter lab` in order to launch Jupyter.

In order to build a Docker image you should be on a computer with _x86_ architecture (not on Mac or other arm-based processor-based machines) because the base layer has only been built for _x86_ machines. You can run `docker build -t zoomcamp-midterm-ak .` in order to build Docker image locally. Then you can create and run Docker container by executing `docker run -p 9696:9696 zoomcamp-midterm-ak`

The webservice is going to be available on the port 9696. Please use [web_client.ipynb](web_client.ipynb) notebook to interact with the webservice.

