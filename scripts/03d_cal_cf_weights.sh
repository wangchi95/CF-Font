for option in seen unseen
do

    item=180
    k=240
    
    basis_len=10
    img_size=80
    basis_n=10
    temperature=0.01

    #model_base="B0_K240BS8x4I1000E200_LR1e-4_Pytorch181_20220423-163142"
    model_base=B0_K240BS32I1000E200_LR1e-4-wdl0.01_20230426-233306
    model=output/models/logs/${model_base}/model_${item}.ckpt

    basis_fn=B0_K240BS32I1000E200_LR1e-4-wdl0.01_20230426-233306_180_basis_240_id_10


    if [ $option == 'seen' ];then
        font_len=240
        # data=data/imgs/Seen240_S80F50_TRAIN800
        # basis_data=data/imgs/BASIS_S80F50_TRAIN800
        data=data/imgs/Seen240_S80F50_FS16
        save_fn=basis/${basis_fn}_ws_${font_len}x10_t${temperature}.pth
    elif [ $option == 'unseen' ]; then
        font_len=60
        data=data/imgs/Unseen60_S80F50_FS16
        save_fn=basis/${basis_fn}_unseen_ws_${font_len}x10_t${temperature}.pth
    fi
    basis_data=data/imgs/BASIS_S80F50_FS16

    CUDA_VISIBLE_DEVICES=1 python cal_cf_weight.py \
    --img_size ${img_size} \
    --data_path ${data} \
    --basis_path ${basis_data} \
    --output_k ${k} \
    --load_model ${model} \
    --font_len ${font_len} \
    --basis_len ${basis_len} \
    --baseline_idx 0 \
    -t ${temperature} \
    --save_fn ${save_fn}
done