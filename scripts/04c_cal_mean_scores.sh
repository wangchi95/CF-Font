set -x
gid=${1:-"0"}

for option in seen unseen
do
    
    if [ $option == 'seen' ];then
        rst_name=CF_from_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20230426-233306_180_200_top-1_ft10_wdl0.01_lr0.01
        font_len=240
    elif [ $option == 'unseen' ]; then
        rst_name=unseen_CF_from_B0_K240BS32I1000E200_LR1e-4-wdl0.01_20230426-233306_180_200_top-1_ft10_wdl0.01_lr0.01
        font_len=60
    fi
    pred_path=output/test_rsts/${rst_name}
    CUDA_VISIBLE_DEVICES=${gid} python eval/cal_mean.py \
    -f ${pred_path}/a_scores/ \
    -k ${font_len}
    # -j 1 3 4 ## !!! jump some fonts like basis font
done