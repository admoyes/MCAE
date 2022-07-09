#!/bin/bash

# build the docker image
docker build -t mcae:latest .


run_training() {
	# PARAMS
	# $1 : run name for this mlflow run
	# $2 : model type (stanosa, mcae, dcae)
	# $3 : domain index 1 (used for dcae and stanosa)
	# $4 : domain index 2 (used for dcae)
	mlflow run . \
		-e main \
		--experiment-name training \
		--run-name $1 \
		--docker-args gpus=all \
		--docker-args shm-size=2G \
		-P model-type=$2 \
		-P domain-index-1=$3 \
		-P domain-index-2=$4 \
		-P epochs=50
}


#################################################################
# STANOSA Training
#################################################################

# train StaNoSA with domain A (0th index)
run_training stanosa_a stanosa 0
#mlflow run . \
#	-e main \
#	--experiment-name training \
#	--run-name stanosa_a \
#	--docker-args gpus=all \
#	--docker-args shm-size=2G \
#	-P model-type=stanosa \
#	-P stanosa-domain-index=0 \
#	-P epochs=50
	

# train StaNoSA with domain B (1st index)
run_training stanosa_b stanosa 1
#mlflow run . \
#	-e main \
#	--experiment-name training \
#	--run-name stanosa_b \
#	--docker-args gpus=all \
#	--docker-args shm-size=2G \
#	-P model-type=stanosa \
#	-P stanosa-domain-index=1 \
#	-P epochs=50


# train StaNoSA with domain C (2nd index)
run_training stanosa_c stanosa 2
#mlflow run . \
#	-e main \
#	--experiment-name training \
#	--run-name stanosa_c \
#	--docker-args gpus=all \
#	--docker-args shm-size=2G \
#	-P model-type=stanosa \
#	-P stanosa-domain-index=2 \
#	-P epochs=50

#################################################################
# DCAE Training
#################################################################

run_training dcae_ab dcae 0 1
#mlflow run . \
#	-e main \
#	--experiment-name training \
#	--run-name dcae_ab \
#	--docker-args gpus=all \
#	--docker-args shm-size=2G \
#	-P model-type=dcae \
#	-P epochs=50 \
#	-P dcae-domain-index-1=0 \
#	-P dcae-domain-index-2=1 \
#
#
run_training dcae_ac dcae 0 2
#mlflow run . \
#	-e main \
#	--experiment-name training \
#	--run-name dcae_ac \
#	--docker-args gpus=all \
#	--docker-args shm-size=2G \
#	-P model-type=dcae \
#	-P epochs=50 \
#	-P dcae-domain-index-1=0 \
#	-P dcae-domain-index-2=2 \
#
#
run_training dcae_bc dcae 1 2
#mlflow run . \
#	-e main \
#	--experiment-name training \
#	--run-name dcae_bc \
#	--docker-args gpus=all \
#	--docker-args shm-size=2G \
#	-P model-type=dcae \
#	-P epochs=50 \
#	-P dcae-domain-index-1=1 \
#	-P dcae-domain-index-2=2 \




#################################################################
# MCAE Training
#################################################################

# train MCAE
run_training mcae mcae 
#mlflow run . \
#	-e main \
#	--experiment-name training \
#	--run-name mcae \
#	--docker-args gpus=all \
#	--docker-args shm-size=2G \
#	-P model-type=mcae \
#	-P epochs=50

