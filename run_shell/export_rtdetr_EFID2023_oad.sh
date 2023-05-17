python3 tools/export_model.py \
-c configs/rtdetr/rtdetr_r50vd_6x_EFID2023_oad.yml \
-o weights=/works/PaddleDetection_oad/output/rtdetr_r50vd_6x_EFID2023_oad/best_model.pdparams trt=True exclude_nms=True
