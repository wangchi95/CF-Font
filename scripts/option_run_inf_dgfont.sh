set -x

item=180 # 200
style_font_len=240

# Make data for DG-Font (Copy the content font to $data/id_${font_len})
# The data file tree like follows: (few-shot 16, target content 5646)
# 
# .
# ├── data
# │   ├── fonts
# │   └── imgs
# │       ├── Seen240_S80F50_FS16_DGFONT
# │       |   ├── id_0
# │       │   │   ├── 0000.png
# │       │   │   ├── 0001.png
# │       │   │   ├── ...
# │       │   │   └── 0015.png
# │       |   ├── id_1
# │       |   ├── ...
# │       |   ├── id_239
# │       │   │   ├── 0000.png
# │       │   │   ├── 0001.png
# │       │   │   ├── ...
# │       │   │   └── 0015.png
# │       |   └── id_240
# │       │       ├── 0000.png
# │       │       ├── 0001.png
# │       │       ├── ...
# │       │       └── 5645.png
# │       └── ...
# ├── charset
# └── ...
# here is a example: (in folder `data/imgs`)
# > cp -rf Seen240_S80F50_FS16 Seen240_S80F50_FS16_DGFONT
# > cp -rf Seen240_S80F50_TEST5646/id_0 Seen240_S80F50_FS16_DGFONT/id_240

output_k=240
py_file=inf_with_style_ft.py
img_size=80
model_base=B0_K240BS32I1000E200_LR1e-4-wdl0.01_20230426-233306

data=data/imgs/Seen240_S80F50_FS16_DGFONT
model=output/models/logs/${model_base}/model_${item}.ckpt
save_path="output/test_rsts/dgfont_${model_base}_${item}"

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 \
    --use_env --master_port 34544 ${py_file} \
    --img_size ${img_size} \
    --data_path ${data} \
    --output_k ${output_k} \
    --load_model ${model} \
    --save_path ${save_path} \
    --font_len ${style_font_len} \
    --baseline_idx 0 \
    --sty_batch_size 40
