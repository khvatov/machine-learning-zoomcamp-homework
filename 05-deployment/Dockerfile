FROM python:3.11-slim

RUN pip install --no-cache-dir pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]