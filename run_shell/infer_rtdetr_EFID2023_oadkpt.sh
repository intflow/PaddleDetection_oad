#!/bin/bash

source /opt/conda/bin/activate
conda activate base

export CUDA_VISIBLE_DEVICES=4,5
python3 -m paddle.distributed.launch --gpus 4,5 tools/infer.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_EFID2023_oadkpt.yml \
    -o weights=output/rtdetr_r50vd_6x_EFID2023_oadkpt/best_model.pdparams \
    --infer_dir=/DL_data_super_ssd/EFID2023/new_hallway/val/image
