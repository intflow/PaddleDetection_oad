#!/bin/bash

source /opt/conda/bin/activate
conda activate base

export CUDA_VISIBLE_DEVICES=4
python3 -m paddle.distributed.launch --gpus 4 tools/infer.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_EFID2023_oad.yml \
    -o weights=output/rtdetr_r50vd_6x_EFID2023_oad/best_model.pdparams \
    --infer_dir=/DL_data_super_ssd/EFID2023/new_hallway/val/image
