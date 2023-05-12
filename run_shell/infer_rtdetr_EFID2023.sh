#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
python3 -m paddle.distributed.launch --gpus 0,1 tools/infer.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_oad.yml \
    -o weights=output/rtdetr_r50vd_6x_oad/best_model.pdparams \
    --infer_dir=/DL_data_super_ssd/EFID2023/new_hallway/val/image
