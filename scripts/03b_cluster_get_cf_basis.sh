n_basis=10
model_name=B0_K240BS32I1000E200_LR1e-4-wdl0.01_20230426-233306
item=180
content_fm=output/embeddings/embedding_${model_name}_${item}/c_src.pth

CUDA_VISIBLE_DEVICES=0 python scripts/basis/get_basis_simple.py \
-c ${content_fm} -lbs 10 -nb ${n_basis} -m ${model_name}_${item}