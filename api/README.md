# Tamil Tech API

A Flask based API for Speech To Text using Tamil Tech library for Chol app.

## Dockerfile

Dockerfile for building docker image with NVidia CUDA and CuDNN support nad to run the api on the docker created image/container

## docker-compose.yml

docker compose file for running the docker container using docker-compose

To run the API as a docker container

`docker-compose --compatibility up`

To run the API as a docker container in background

`docker-compose --compatibility up -d`

## app.ini

uWSGI config file for running the Flask App via WSGI

## app.py

The Flask app file that uses tamil tech's zero shot module for ASR

## requirements.txt

Requirements file of python libraries required for the backend API