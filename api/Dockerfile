# FROM python:3.7.3-stretch
# WORKDIR /app
# ADD . /app
# RUN apt-get update
# RUN yes | apt-get install libsndfile1 libsndfile1-dev
# RUN pip install -U pip && pip install -r requirements.txt
# EXPOSE 5000
# CMD ["uwsgi", "app.ini"]
FROM nvidia/cuda:11.1-devel-ubuntu18.04 
#11.1-runtime-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV CUDNN_VERSION 8.0.4.30

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.1 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/inference/api
WORKDIR /home/inference/api
COPY . /home/inference/api

RUN apt-get update
RUN yes | apt-get install libsndfile1 libsndfile1-dev python3 python3-dev python3-pip
RUN pip3 install -U pip && pip3 install -r requirements.txt

# CMD ["uwsgi", "app.ini"]
CMD ["python3", "app.py"]