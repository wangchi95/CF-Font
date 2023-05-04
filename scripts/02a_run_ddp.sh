mkdir output
mkdir output/models

CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch \
    --nproc_per_node=2 --use_env main.py \
    --img_size 80 \
    --data_path data/imgs/Seen240_S80F50_TRAIN800 \
    --lr 1e-4 \
    --output_k 240 \
    --batch_size 16 \
    --iters 1000 \
    --epoch 200 \
    --val_num 10 \
    --baseline_idx 0 \
    --save_path output/models \
    --model_name B0_K240BS32I1000E200_LR1e-4-wdl0.01 \
    --ddp \
    --wdl --w_wdl 0.01 \
    --no_val
    # --load_model CF-Font/output/models/logs/B0_K240BS32I1000E200_LR1e-4-wdl0.01_20230426-233306