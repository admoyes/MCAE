#!/bin/bash

# build the docker image
docker build -t mcae:latest .

# run the current project (see MLproject file)
mlflow run . \
	--docker-args gpus=all \
	--docker-args shm-size=2G
