for font_set in Seen240_S80F50 Unseen60_S80F50
do
    full_fp=data/imgs/${font_set}_FULL
    train_fp=data/imgs/${font_set}_TRAIN800
    test_fp=data/imgs/${font_set}_TEST5646
    fewshot_fp=data/imgs/${font_set}_FS16

    full_ch=charset/GB2312_CN6763.txt
    train_ch=charset/TRAIN800.txt
    test_ch=charset/TEST5646.txt
    fewshot_ch=charset/FS16.txt # in train_ch

    python scripts/data_preparation/gen_subset.py  -i ${full_fp} -o ${train_fp} -ic ${full_ch} -oc ${train_ch}
    python scripts/data_preparation/gen_subset.py  -i ${full_fp} -o ${test_fp} -ic ${full_ch} -oc ${test_ch}
    python scripts/data_preparation/gen_subset.py  -i ${full_fp} -o ${fewshot_fp} -ic ${full_ch} -oc ${fewshot_ch}
done