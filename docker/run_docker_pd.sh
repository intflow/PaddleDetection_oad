#!/bin/bash
#X11

#sudo docker login docker.io -u kmjeon -p # Type yourself

#sudo bash attach_NAS_ftp.sh
sudo xhost +local:root

#Mount Data folders
#sudo mkdir /DL_data_big
#sudo mount -t cifs //192.168.0.18/DL_data_big /DL_data_big -o username=intflow2,password=intflow3121!
#sudo mkdir /DL_data_super_ssd
#sudo mount -t nfs 192.168.0.20:/volume1/DL_data_super_ssd /DL_data_super_ssd
#sudo mkdir /DL_data_super_hdd
#sudo mount -t nfs 192.168.0.20:/volume2/DL_data_super_hdd /DL_data_super_hdd
#sudo mount 192.168.0.18:volume1/DL_data_big /DL_data_big
#sudo mount 192.168.0.14:/NAS1 /NAS1

#Pull update docker image
docker pull intflow/paddle_detection:TensorRT8.2.1
#docker pull nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu18.04

#Run Dockers for YOLOXOAD
sudo docker run --name paddle_detection \
--gpus all --rm -p 6436:6436 \
--mount type=bind,src=/home/intflow/works,dst=/works \
--mount type=bind,src=/DL_data_big,dst=/data \
--mount type=bind,src=/DL_data,dst=/DL_data \
--mount type=bind,src=/DL_data_super_ssd,dst=/DL_data_super_ssd \
--mount type=bind,src=/DL_data_super_hdd,dst=/DL_data_super_hdd \
--net=host \
--privileged \
--ipc=host \
-it intflow/paddle_detection:TensorRT8.2.1 /bin/bash

#-it nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu18.04 /bin/bash






