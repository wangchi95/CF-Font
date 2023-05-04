set -x

size=80
item=180
k=240
basis_n=10
data=data/imgs/Seen240_S80F50_TRAIN800
model_base=B0_K240BS32I1000E200_LR1e-4-wdl0.01_20230426-233306
model_name=CF_from_${model_base}_${item}
base_idxs="basis/${model_base}_${item}_basis_${k}_id_${basis_n}.txt"
base_ws="basis/${model_base}_${item}_basis_${k}_id_${basis_n}_ws_${k}x${basis_n}_t0.01.pth"

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 --use_env main.py \
    --content_fusion \
    --img_size ${size} \
    --data_path ${data} \
    --lr 1e-4 \
    --output_k ${k} \
    --batch_size 16 \
    --iters 1000 \
    --epoch 200 \
    --val_num 10 \
    --baseline_idx 0 \
    --save_path output/models \
    --load_model ${model_name} \
    --base_idxs ${base_idxs} --base_ws ${base_ws} \
    --ddp \
    --no_val \
    --wdl --w_wdl 0.01