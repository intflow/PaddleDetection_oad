paddle2onnx --model_dir=./output_inference/rtdetr_r50vd_6x_EFID2023_oad \
            --model_filename "model.pdmodel"  \
            --params_filename "model.pdiparams" \
            --opset_version 16 \
            --save_file output_onnx/rtdetr_r50vd_6x_EFID2023_oad.onnx
