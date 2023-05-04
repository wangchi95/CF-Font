import argparse
import os
import shutil
import pandas as pd
import cv2
import tqdm
import lpips
import time

from tqdm.cli import main

from eval_utils import LPIPS, L1, RMSE, SSIM

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-gt','--dir_gt', type=str, default='./imgs/ex_dir0')
parser.add_argument('-pred','--dir_pred', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists')
parser.add_argument('-m','--methods', type=str, default='', nargs='+')
parser.add_argument('--subfolder', action='store_true')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--standalone', action='store_true', help='standalone metric keep alone in output file')
parser.add_argument('--check', action='store_true', help='check empty file')
parser.add_argument('-t','--tmp_folder', type=str, default='/tmp', help='tmp folder for check empty file')

opt = parser.parse_args()

methods_choices = ['l1', 'rmse', 'ssim', 'lpips', 'fid']
methods_standalone = ['fid']
if opt.methods is '':
	use_methods = methods_choices
else:
	use_methods = opt.methods
	assert all([mi in methods_choices for mi in use_methods]), 'invalid mathods exist'

def load_image(path):
	assert path[-3:]=='bmp' or path[-3:]=='jpg' or path[-3:]=='png' or path[-4:]=='jpeg'
	return cv2.imread(path)[:,:,::-1]

def easy_log(msg, flag=None, fn=None, append=False, on_screen=True, pd_df=None):
	if flag == '=':
		msg += '\n======================='
	elif flag == '-':
		msg += '\n-------------'
	elif flag == '!':
		msg += '!!!\n!!!\n'
	
	if on_screen:
		print(msg)
		if pd_df is not None: print(pd_df)
	if fn:
		with open(fn, 'a' if append else 'w') as f:
			msg = '{:}\n{:}\n'.format(msg, pd_df.to_string()) if pd_df is not None else msg+'\n'
			f.write(msg)


if __name__ == '__main__':
	methods_allinone = set(use_methods) - set(methods_standalone)
	methods_standalone = set(use_methods) - methods_allinone

	files = sorted(os.listdir(opt.dir_gt))
	for file in files:
		assert os.path.exists(os.path.join(opt.dir_pred,file))

	empty_files = []
	remain_files = []

	exist_empty_file = False
	if opt.check:
		for file in tqdm.tqdm(files):
			img = cv2.imread(os.path.join(opt.dir_gt,file))
			if img.mean() == 255: # all white
				empty_files.append(file)
			else:
				remain_files.append(file)
		exist_empty_file = len(empty_files) > 0
		if exist_empty_file:
			easy_log('Finding {} empty imgs: {}'.format(len(empty_files), empty_files), flag = '!', fn = opt.out + '_scores.txt')
			files = remain_files
			gt_true, pred_true = opt.dir_gt, opt.dir_pred
			gt_fake, pred_fake = os.path.basename(opt.dir_gt), os.path.basename(opt.dir_pred)
			if gt_fake == pred_fake: gt_fake = 'gt_' + gt_fake
			gt_fake, pred_fake = os.path.join(opt.tmp_folder, gt_fake), os.path.join(opt.tmp_folder, pred_fake)
			opt.dir_gt, opt.dir_pred = gt_fake, pred_fake
			if os.path.exists(opt.dir_gt) or os.path.exists(opt.dir_pred):
				ans = input(f'Delete <2{gt_fake}> and <{pred_fake}>? [y/n] ')
				assert ans == 'y'
				os.system(f'rm -rf {gt_fake} & rm -rf {pred_fake}')
			assert not os.path.exists(opt.dir_gt) and not os.path.exists(opt.dir_pred)
			os.mkdir(opt.dir_gt)
			os.mkdir(opt.dir_pred)
			for file in files:
				shutil.copy(os.path.join(gt_true, file), os.path.join(opt.dir_gt, file))
				shutil.copy(os.path.join(pred_true, file), os.path.join(opt.dir_pred, file))



	if len(methods_standalone) > 0:
		easy_log('Standalone Metrics', flag = '=', fn = opt.out + '_scores.txt', append=len(empty_files) > -1)
		# Fid
		if 'fid' in use_methods:
			suffix = '_fid.txt' if opt.standalone else '_scores.txt'
			st = time.time()
			print('Calculate Fid...')
			print('writing to txt:', opt.out + suffix)
			os.system(f'python -m pytorch_fid {opt.dir_gt} {opt.dir_pred} --batch-size 100 | tee -a {opt.out}{suffix}') # 2>&1 | tee
			print('done! using {:.2f}s'.format(time.time() - st))

	# ALL in one
	if len(methods_allinone) > 0:
		if 'lpips' in use_methods:
			lpips_model = LPIPS(using_gpu=opt.use_gpu)
			
		easy_log('\nAllInOne Metrics', flag = '=', fn = opt.out + '_scores.txt', append=True)

		rsts = {}
		for i in use_methods:
			if i != 'fid':
				rsts[i] = []
		num = len(files)

		st = time.time()
		fns = []
		for file in tqdm.tqdm(files):
			if(os.path.exists(os.path.join(opt.dir_pred,file))):
				# Load images
				fns.append(file)
				img_gt = lpips.load_image(os.path.join(opt.dir_gt,file)) # HWC, RGB, [0, 255]
				img_pred = lpips.load_image(os.path.join(opt.dir_pred,file)) # HWC, RGB, [0, 255]
				if 'lpips' in use_methods:
					rst_lpips = lpips_model.cal_lpips(img_gt, img_pred)
					rsts['lpips'].append(rst_lpips.detach().cpu().numpy())
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

		easy_log('Mean', flag = '-', fn = opt.out + '_scores.txt', append=True, pd_df=tab.mean(0))
		print('-------------\n')
		easy_log('Detail', flag = '-', fn = opt.out + '_scores.txt', append=True, pd_df=tab, on_screen=False)

		print('writing to txt:', opt.out + '_scores.txt')
		print('done! using {:.2f}s'.format(time.time() - st))

		if exist_empty_file:
			os.system(f'rm -rf {gt_fake} & rm -rf {pred_fake}')