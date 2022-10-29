#!/bin/bash

DOCKER_BUILDKIT=1 docker build --tag fastwhisper:latest app/

MODEL_NAME=tiny
DIR=app/code/
FILE=$DIR/$MODEL_NAME.onnx
if [[ -f "$FILE" ]]; then
    echo "$FILE is already created."
else
    echo "Exporting $MODEL_NAME model to ONNX."
    docker run -v $PWD/$DIR:/opt/project/ --shm-size=4gb --env MODEL_NAME=$MODEL_NAME fastwhisper python3 export_onnx_model.py
fi
FILE=$DIR/${MODEL_NAME}_fp16.engine
if [[ -f "$FILE" ]]; then
    echo "$FILE is already created."
else
    echo "Converting $MODEL_NAME ONNX model to TensorRT."
    docker run -v $PWD/$DIR:/opt/project/ --shm-size=4gb fastwhisper trtexec --onnx=$MODEL_NAME.onnx --fp16 --saveEngine=${MODEL_NAME}_fp16.engine --buildOnly --memPoolSize=workspace:4000
fi


# start the app
docker-compose up