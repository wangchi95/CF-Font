import argparse
import os
import torch
import pandas as pd
import numpy as np
import cv2
import glob
import tqdm
import lpips
import time

import pdb

from eval_utils import LPIPS, L1, RMSE, SSIM

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-gt','--dir_gt', type=str, default='../test/data_writing_unseen_10_test_200')
parser.add_argument('-pred','--dir_pred', type=str, default='../test/tmp_base')
parser.add_argument('-o','--out', type=str, default='a_scores')
parser.add_argument('-m','--methods', type=str, default='', nargs='+')
parser.add_argument('--subfolder', action='store_true')
parser.add_argument('--gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('-j','--jump', type=int, default=[], nargs='+')
parser.add_argument('--only', type=int, default=[], nargs='+')

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
	lpips_model = LPIPS(using_gpu=opt.gpu)


# crawl directories
# f = open(opt.out,'w')

folders = sorted(os.listdir(opt.dir_gt), key=lambda k: int(k.lstrip('id_')))
opt.out = os.path.join(opt.dir_pred, opt.out)
print(opt.out)
# exit()
if not os.path.exists(opt.out):
	os.mkdir(opt.out)

rsts_all = {i:[] for i in use_methods if i != 'fid'}

fs_bar = tqdm.tqdm(folders)
for folder in fs_bar:
	if int(folder.lstrip('id_')) in opt.jump:
		continue
	if int(folder.lstrip('id_')) in opt.only or len(opt.only) == 0:
		dir_gt = os.path.join(opt.dir_gt, folder)
		if not os.path.isdir(dir_gt): continue
		dir_pred = os.path.join(opt.dir_pred, folder)

		# rename
		fns = glob.glob(os.path.join(dir_pred, '*png'))
		for fn in fns:
			new_base_fn = '{:04}.png'.format(int(os.path.basename(fn).split('.')[0]))
			# print(fn, new_base_fn)
			os.rename(fn, os.path.join(dir_pred, new_base_fn))
		# exit()

		out = os.path.join(opt.out, folder)

		files = sorted(os.listdir(dir_gt))
		rsts = {i:[] for i in use_methods if i != 'fid'}
		num = len(files)

		st = time.time()
		fns = []
		for file_g in tqdm.tqdm(files, leave=False):
			if not file_g.endswith('.png'): continue
			file_p = file_g
			assert os.path.exists(os.path.join(dir_pred,file_p))
			# Load images
			fns.append(file_g)
			img_gt = lpips.load_image(os.path.join(dir_gt,file_g)) # HWC, RGB, [0, 255]
			img_pred = lpips.load_image(os.path.join(dir_pred,file_p)) # HWC, RGB, [0, 255]
			
			if 'lpips' in use_methods:
				rst_lpips = lpips_model.cal_lpips(img_gt, img_pred)
				rsts['lpips'].append(rst_lpips.cpu().detach().numpy())
				rsts_all['lpips'].append(rst_lpips.cpu().detach().numpy())
			if 'l1' in use_methods:
				rst_l1 = L1(img_gt, img_pred, 255.)
				rsts['l1'].append(rst_l1)
				rsts_all['l1'].append(rst_l1)
			if 'rmse' in use_methods:
				rst_rmse = RMSE(img_gt, img_pred, 255.)
				rsts['rmse'].append(rst_rmse)
				rsts_all['rmse'].append(rst_rmse)
			if 'ssim' in use_methods:
				rst_ssim = SSIM(img_gt, img_pred)
				rsts['ssim'].append(rst_ssim)
				rsts_all['ssim'].append(rst_ssim)

		tab = pd.DataFrame(rsts, fns)

		# print('Mean')
		# print('=============')
		# pdb.set_trace()
		# tm = tab.mean(0)
		# info = list(zip(tm.index, tm.to_numpy()))
		fs_bar.set_postfix({k:np.mean(v) for k,v in rsts_all.items()})
		# print(tab.mean(0))

		# print('writing to txt:', out + '_scores.txt')
		with open(out + '_scores.txt', 'w') as f:
			f.write('Mean\n')
			f.write('=============\n')
			f.write(tab.mean().to_string())
			f.write('\n\nDetail\n')
			f.write('=============\n')
			f.write(tab.to_string())
		# print(f'done! using {time.time() - st}s')

		if 'fid' in use_methods:
			st = time.time()
			# print('Calculate Fid...')
			os.system(f'python -m pytorch_fid {dir_gt} {dir_pred} 1>{out}_fid.txt 2>/dev/null') #  2>&1
			# print(f'done! using {time.time() - st}s')

if opt.only == 0:
	with open(os.path.join(opt.out, 'all_scores.txt'), 'w') as f:
		f.write('Mean\n')
		f.write('=============\n')
		for k, v in rsts_all.items():
			print(k, np.mean(v))
			f.write(f'{k}: {np.mean(v)}')
