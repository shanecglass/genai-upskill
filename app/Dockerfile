# Use an official lightweight Python image.
FROM python:3.12.7-slim

RUN apt-get update && \
  apt-get -y install git \
  locales \
  locales-all && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Install dependencies into this container so there's no need to
# install anything at container run time.
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir ./srv/app

# Create non-root user
RUN groupadd -g 900 mesop && useradd -u 900 -s /bin/bash -g mesop mesop
USER mesop

# Copy local code to the container image.
WORKDIR /home/ubuntudocker/srv/app

COPY main.py .
COPY modules.py .
COPY model_calls.py .
COPY ./model_mgmt /home/ubuntudocker/srv/app/model_mgmt


# Service must listen to $PORT environment variable.
# This default value facilitates local development.
ENV PORT 8080

# Run the web service on container startup. Here you use the gunicorn
# server, with one worker process and 8 threads. For environments
# with multiple CPU cores, increase the number of workers to match
# the number of cores available.
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 main:me
