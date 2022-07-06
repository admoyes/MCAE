#!/bin/bash

# build the docker image
docker build -t mcae:latest .

#################################################################
# DCAE Training
#################################################################

mlflow run . \
	-e main \
	--experiment-name training \
	--run-name dcae_ab \
	--docker-args gpus=all \
	--docker-args shm-size=2G \
	-P model-type=dcae \
	-P epochs=1 \
	-P dcae-domain-index-1=0 \
	-P dcae-domain-index-2=1 \


mlflow run . \
	-e main \
	--experiment-name training \
	--run-name dcae_ac \
	--docker-args gpus=all \
	--docker-args shm-size=2G \
	-P model-type=dcae \
	-P epochs=1 \
	-P dcae-domain-index-1=0 \
	-P dcae-domain-index-2=2 \


mlflow run . \
	-e main \
	--experiment-name training \
	--run-name dcae_bc \
	--docker-args gpus=all \
	--docker-args shm-size=2G \
	-P model-type=dcae \
	-P epochs=1 \
	-P dcae-domain-index-1=1 \
	-P dcae-domain-index-2=2 \


#################################################################
# STANOSA Training
#################################################################

# train StaNoSA with domain A (0th index)
mlflow run . \
	-e main \
	--experiment-name training \
	--run-name stanosa_a \
	--docker-args gpus=all \
	--docker-args shm-size=2G \
	-P model-type=stanosa \
	-P stanosa-domain-index=0 \
	-P epochs=1
	

# train StaNoSA with domain B (1st index)
mlflow run . \
	-e main \
	--experiment-name training \
	--run-name stanosa_b \
	--docker-args gpus=all \
	--docker-args shm-size=2G \
	-P model-type=stanosa \
	-P stanosa-domain-index=1 \
	-P epochs=1


# train StaNoSA with domain C (2nd index)
mlflow run . \
	-e main \
	--experiment-name training \
	--run-name stanosa_c \
	--docker-args gpus=all \
	--docker-args shm-size=2G \
	-P model-type=stanosa \
	-P stanosa-domain-index=2 \
	-P epochs=1



#################################################################
# MCAE Training
#################################################################

# train MCAE
mlflow run . \
	-e main \
	--experiment-name training \
	--run-name mcae \
	--docker-args gpus=all \
	--docker-args shm-size=2G \
	-P model-type=mcae \
	-P epochs=1

