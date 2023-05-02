#!/bin/bash

#sudo docker login docker.io -u kmjeon -p # Type yourself

sudo docker commit yolov7_oad paddle_detection:TensorRT8.2.1
sudo docker tag paddle_detection:TensorRT8.2.1 intflow/paddle_detection:TensorRT8.2.1
sudo docker push intflow/paddle_detection:TensorRT8.2.1
