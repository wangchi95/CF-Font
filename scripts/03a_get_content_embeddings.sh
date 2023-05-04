k=240
item=180
img_size=80
model_base=B0_K240BS32I1000E200_LR1e-4-wdl0.01_20230426-233306
model=output/models/logs/${model_base}/model_${item}.ckpt

# data=data/imgs/Seen240_S80F50_TRAIN800
# !!! The more characters used to cluster, the better. 
# However, since the memory limitation, you can also use a subset of data_K240_S80F50_TRAIN800 to cluster.
# like random choose 50 characters or simply use few-shot 16 characters.
data=data/imgs/Seen240_S80F50_FS16

CUDA_VISIBLE_DEVICES=0 python collect_content_embeddings.py --img_size ${img_size} \
--data_path ${data} \
--output_k ${k} \
--batch_size 32 \
--load_model ${model} \
--save_path output/embeddings/embedding_${model_base}_${item} \
--baseline_idx 0 \
--n_atts ${k} \
--no_skip