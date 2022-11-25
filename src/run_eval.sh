#!/bin/bash

# build the docker image
docker build -t mcae:latest .

run_eval() {
	# PARAMS
	# $1 : run name for this mlflow run
	# $2 : run name used when training the auto encoder model
	# $3 : model type (stanosa, mcae, dcae)
	# $4 : domain index (used to route data to specific auto encoders inside a mcae or dcae model)
	for dataset in kather colon lung
	do
		for clf in mlp xgboost svc 
		do
			mlflow run . \
				-e eval \
				--experiment-name eval \
				--run-name $1 \
				--docker-args gpus=all \
				--docker-args shm-size=2G \
				-P model-type=$3 \
				-P training-experiment-name=training \
				-P training-run-name=$2 \
				-P classifier=$clf \
				-P domain-index=$4 \
				-P dataset-path=/data/$dataset
		done
	done
}


#################################################################
# StaNoSA Eval
#################################################################
run_eval stanosa_a stanosa_a stanosa
run_eval stanosa_b stanosa_b stanosa
run_eval stanosa_c stanosa_c stanosa


#################################################################
# DCAE Eval
#################################################################
run_eval dcae_ab_a dcae_ab dcae 0
run_eval dcae_ac_a dcae_ac dcae 0
run_eval dcae_bc_b dcae_bc dcae 0
run_eval dcae_ab_b dcae_ab dcae 1
run_eval dcae_ac_c dcae_ac dcae 1
run_eval dcae_bc_c dcae_bc dcae 1


#################################################################
# MCAE Eval
#################################################################
run_eval mcae_a mcae mcae 0
run_eval mcae_b mcae mcae 1
run_eval mcae_c mcae mcae 2
