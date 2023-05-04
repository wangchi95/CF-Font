import os
import cv2
import numpy as np

import glob
from tqdm import tqdm

# n = 400
# path = 'data_400_hard_50'
# out_p = 'data_400_hard_50_2x2_format'

n = 221
path = '../../dataset/Data_K221/data_K221_S128F80_Train800_Base50/'
out_p = 'data_221_S128F80_Base50_2x2_format'

os.makedirs(out_p, exist_ok=True)

for i in tqdm(range(n)):
    folder = os.path.join(path, f'id_{i}')
    img_out_p = os.path.join(out_p, '{:04}.png'.format(i))
    img_fns = glob.glob(os.path.join(folder, '*.png'))
    imgs = [cv2.imread(img_fn) for img_fn in sorted(img_fns)] # 50
    # hw = int(len(imgs) ** 0.5)
    hw = 2
    ih, iw, _ = imgs[0].shape
    imgs = imgs[:hw*hw]
    imgs = np.array(imgs).reshape((hw,hw,ih,iw,3)).transpose((0,2,1,3,4)).reshape((hw*ih,hw*iw,3))
    cv2.imwrite(img_out_p, imgs)
