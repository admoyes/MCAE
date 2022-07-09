FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel
RUN apt update
RUN apt install git -y
RUN pip install \
	scikit-image \
	einops \
	pillow \
	numpy \
	mlflow \
	umap-learn \
	matplotlib \
	tqdm \
	xgboost
