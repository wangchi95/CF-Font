import argparse
import os
import torch
import pandas as pd
import cv2
import tqdm
import lpips
import time

from .eval_utils import LPIPS, L1, RMSE, SSIM

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-gt','--dir_gt', type=str, default='../test/data_writing_unseen_10_test_200')
parser.add_argument('-pred','--dir_pred', type=str, default='../test/tmp_base')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists')
parser.add_argument('-m','--methods', type=str, default='', nargs='+')
parser.add_argument('--subfolder', action='store_true')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

methods_choices = ['l1', 'rmse', 'ssim', 'lpips', 'fid']
if opt.methods is '':
    use_methods = methods_choices
else:
    use_methods = opt.methods
    assert all([mi in methods_choices for mi in use_methods]), 'invalid mathods exist'

def load_image(path):
    assert path[-3:]=='bmp' or path[-3:]=='jpg' or path[-3:]=='png' or path[-4:]=='jpeg'
    return cv2.imread(path)[:,:,::-1]

## Initializing the model
if 'lpips' in use_methods:
    lpips_model = LPIPS(using_gpu=opt.use_gpu)


# crawl directories
# f = open(opt.out,'w')

files = os.listdir(opt.dir_gt)

rsts = {}
for i in use_methods:
    if i != 'fid':
        rsts[i] = []
num = len(files)

st = time.time()
fns = []
for file in tqdm.tqdm(files):
    if not file.endswith('.png'): continue
    assert os.path.exists(os.path.join(opt.dir_gt,file))
    # Load images
    fns.append(file)
    img_gt = lpips.load_image(os.path.join(opt.dir_gt,file)) # HWC, RGB, [0, 255]
    img_pred = lpips.load_image(os.path.join(opt.dir_pred,file)) # HWC, RGB, [0, 255]
    if 'lpips' in use_methods:
        rst_lpips = lpips_model.cal_lpips(img_gt, img_pred)
        rsts['lpips'].append(rst_lpips)
    if 'l1' in use_methods:
        rst_l1 = L1(img_gt, img_pred, 255.)
        rsts['l1'].append(rst_l1)
    if 'rmse' in use_methods:
        rst_rmse = RMSE(img_gt, img_pred, 255.)
        rsts['rmse'].append(rst_rmse)
    if 'ssim' in use_methods:
        rst_ssim = SSIM(img_gt, img_pred, 255.)
        rsts['ssim'].append(rst_ssim)

tab = pd.DataFrame(rsts, fns)

print('Mean')
print('=============')
print(tab.mean(0))

print('writing to txt:', opt.out + '_scores.txt')
with open(opt.out + '_scores.txt', 'w') as f:
    f.write('Mean\n')
    f.write('=============\n')
    f.write(tab.mean().to_string())
    f.write('\n\nDetail\n')
    f.write('=============\n')
    f.write(tab.to_string())
print(f'done! using {time.time() - st}s')

if 'fid' in use_methods:
    st = time.time()
    print('Calculate Fid...')
    os.system(f'python -m pytorch_fid --batch-size 25 --device cuda:1 {opt.dir_gt} {opt.dir_pred} 1>{opt.out}_fid.txt 2>&1')
    print(f'done! using {time.time() - st}s')
