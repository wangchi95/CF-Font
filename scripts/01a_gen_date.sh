img_size=80 # 128
chara_size=50 # 80
chara=charset/GB2312_CN6763.txt  #charset/test.txt 

font_basefolder=data/fonts
out_basefolder=data/imgs
mkdir $out_folder

for font_set in Seen240 Unseen60
do
    font_folder=${font_basefolder}/Font_$font_set
    out_folder=${out_basefolder}/${font_set}_S${img_size}F${chara_size}_FULL
    mkdir $out_folder

    python font2img.py  --ttf_path $font_folder \
                        --img_size $img_size \
                        --chara_size $chara_size \
                        --chara $chara \
                        --save_path $out_folder
done