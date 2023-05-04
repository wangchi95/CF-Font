# CF-Font

-------------
<img width="1339" alt="teaser" src="https://user-images.githubusercontent.com/96471617/227506740-764b4e8f-75f1-47f0-b750-7a32f371281c.png">

By Chi Wang, Min Zhou, Tiezheng Ge, Yuning Jiang, Hujun Bao, Weiwei Xu

This repo is the official implementation of `CF-Font: Content Fusion for Few-shot Font Generation` accepted by CVPR 2023.

## Video demos for Style Interpolation
- A poem demo


https://user-images.githubusercontent.com/96471617/227514150-b9ea651f-3859-489b-a24b-5623d806aca8.mp4



- Comparison with DG-Font



https://user-images.githubusercontent.com/96471617/227696956-10b663ad-a0bd-4759-847d-a61d08eb97bf.mp4



## Dependencies

Libarary
-------------
```
pytorch (>=1.0)
tqdm
numpy
opencv-python  
scipy
sklearn
matplotlib  
pillow  
tensorboardX
scikit-image
scikit-learn
pytorch-fid
lpips
pandas
kornia
```

DCN
--------------

please refer to https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0 to install the dependencies of deformable convolution.

Dataset
--------------
[方正字库](https://www.foundertype.com/index.php/FindFont/index) provides free font download for non-commercial users.

# How to run

1. prepare dataset
    - Put your font files to a folder and character file to charset
        ```bash
        .
        ├── data
        │   └── fonts
        │       ├── Font_Seen240
        │       |   ├── 000_xxxx.ttf
        │       |   ├── 001_xxxx.ttf
        │       |   ├── ...
        │       |   └── 239_xxxx.ttf
        │       └── Font_Unseen60
        ├── charset
        │   ├── check_overlap.py
        │   ├── GB2312_CN6763.txt # all characterx used in TRAIN and TEST
        │   ├── FS16.txt          # Few-Shot characters, should not be included in TESTxxx.txt for fairness (can be included in TRAINxxx.txt) 
        │   ├── TEST5646.txt      # all character used in TRAIN
        │   └── TRAIN800.txt      # all character used in TEST
        └── ...
        ```
    - Generate dataset with the full standard Chinese character set (6763 in total) of GB/T 2312 : 
        ```bash
        sh scripts/01a_gen_date.sh
        ```
        After that, your file tree should be:
        ```bash
        .
        ├── data
        │   ├── fonts
        │   └── imgs
        │       ├── Seen240_S80F50_FULL
        │       |   ├── id_0
        │       │   │   ├── 0000.png
        │       │   │   ├── 0001.png
        │       │   │   ├── ...
        │       │   │   └── 6762.png
        │       |   ├── id_1
        │       |   ├── ...
        │       |   └── id_239
        │       └── Unseen60_S80F50_FULL
        ├── charset
        └── ...
        ```
    - Get subsets with train, test and fewshot character txts.
        ```bash
        sh scripts/01b_copy_subset.sh
        ```
        After that, your file tree should be:
        ```bash
        .
        ├── data
        │   ├── fonts
        │   └── imgs
        │       ├── Seen240_S80F50_FS16
        │       |   ├── id_0
        │       │   │   ├── 0000.png
        │       │   │   ├── 0001.png
        │       │   │   ├── ...
        │       │   │   └── 0015.png
        │       |   ├── id_1
        │       |   ├── ...
        │       |   └── id_239
        │       ├── Seen240_S80F50_FULL
        │       ├── Seen240_S80F50_TEST5646
        │       |   ├── id_0
        │       │   │   ├── 0000.png
        │       │   │   ├── 0001.png
        │       │   │   ├── ...
        │       │   │   └── 5645.png
        │       |   ├── id_1
        │       |   ├── ...
        │       |   └── id_239
        │       ├── Seen240_S80F50_TRAIN800
        │       |   ├── id_0
        │       │   │   ├── 0000.png
        │       │   │   ├── 0001.png
        │       │   │   ├── ...
        │       │   │   └── 0799.png
        │       |   ├── id_1
        │       |   ├── ...
        │       |   └── id_239
        │       ├── Unseen60_S80F50_FS16
        │       ├── Unseen60_S80F50_FULL
        │       ├── Unseen60_S80F50_TEST5646
        │       └── Unseen60_S80F50_TRAIN800
        ├── charset
        └── ...
        ```
2. Train base network
    ```bash
    # enable PC-WDL with the flag `--wdl` and PC-PKL with the flag `-pkl` 
    sh scripts/02a_run_ddp.sh
    ```
    Option: In order to evaluate the training of the network, we can use the script `scripts/option_run_inf_dgfont.sh` to inference.
3. Train CF-Font
    - Select basis. Basis fonts better contain a standard font, like `song`.
        - Manually. If you want select manually, please put basis ids (one line, seperated with a space) to a txt file, like:
            ```
            0 1 2 3 4 5 6 7 8 9
            ```
        - By clustering.
            ```bash
            # Content embeddings collection 
            sh scripts/03a_get_content_embeddings.sh

            # obtain basis ids through clustering
            sh scripts/03b_cluster_get_cf_basis.sh
            ```
    - Get subsets with basis font ids.
        ```bash
        sh scripts/03c_copy_basis_subset.sh
        ```
    - Get basis weight for content fusion:
        ```bash
        sh scripts/03d_cal_cf_weights.sh
        ```
    - Train CF-Font
        ```bash
        # make a folder for CF-Font training, and copy the pretrain model here.
        sh scripts/03e_init_cf_env.sh

        # train CF-Font
        sh scripts/03f_run_ddp_cf.sh
        ```
4. Inference and evaluation:
    ```bash
    # Inference (SII with the flag `--ft`)
    sh scripts/04a_run_inf_cf.sh

    # Evaluation
    ## get scores for each font
    sh scripts/04b_get_scores.sh

    ## get mean scores (use `-j` to skip the unwanted fonts, like basis fonts)
    sh scripts/04c_cal_mean_scores.sh
    ```

## Citation

```
@inproceedings{CF-Font,
    title={CF-Font: Content Fusion for Few-shot Font Generation},
    author={Chi Wang, Min Zhou, Tiezheng Ge, Yuning Jiang, Hujun Bao, Weiwei Xu},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    year={2023}
}
```

## Acknowledgements
We would like to thank [Alimama](https://www.alimama.com) (Alibaba Group) and [State Key Lab of CAD&CG](http://www.cad.zju.edu.cn) (Zhejiang University) for their support and advices in our project. Our code is based on [DG-Font](https://github.com/ecnuycxie/DG-Font).

## Contact

If you have any questions, please feel free to contact `wangchi1995@zju.edu.cn`.