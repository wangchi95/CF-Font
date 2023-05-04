import os
import cv2
import time
import glob
import tqdm
import torch
import argparse
import sklearn
import numpy as np
# from matplotlib import pyplot as plt
from sklearn.cluster import KMeans #, AffinityPropagation, MiniBatchKMeans
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='Get ContentFusion basis')
parser.add_argument('-c', '--content', type=str, default='../../../embedding_baseline/c_src.pth', help='path to content embedding')
parser.add_argument('-m', '--model_name', type=str)
parser.add_argument('-if', '--ignore_font', default=[], type=int, nargs='+', help='the font to drop in basis')
parser.add_argument('-ic', '--ignore_char', default=[], type=int, nargs='+', help='the char to drop in basis')
parser.add_argument('-nb', '--basis_number', default=[10], type=int, nargs='+', help='the number of basis')
parser.add_argument('-lbs', '--load_bs', default=1, type=int, help='the batchsize for cal distance')
args = parser.parse_args()

cvs = torch.load(args.content)#.cpu().numpy()
k, n_samples, _, _, _ = cvs.shape
print(cvs.shape) # (221, 50, 256, 32, 32)

# filter out?
if len(args.ignore_font) == 0 and len(args.ignore_char) == 0:
    n_samples_remain = n_samples
else:
    ignore_font = args.ignore_font
    ignore_char = args.ignore_char
    ignore_char = torch.tensor(ignore_char)
    mask = torch.ones(n_samples, dtype=bool)
    mask.scatter_(0, ignore_char, False)
    n_samples_remain = mask.sum()
    print(f'remain: {n_samples_remain}/{n_samples}')
    cvs = cvs[:, mask]

# get embedding
cvs = cvs.reshape(*cvs.shape[:2], -1) # [221, n_samples_remain, xxx] 
# L1
cv_dis_s = []
per = args.load_bs
assert k%per == 0
for i in tqdm.tqdm(range(k//per)):
    cv_dis = (cvs[:,None,:] - cvs[i*per:(i+1)*per][None,...]).abs().mean(-1) # [221, 1, k] - [1, 20,k] -> [400, 20, k] -> [400,20]
    cv_dis_s.append(cv_dis)

cv_dis = torch.cat(cv_dis_s, 1)
assert cv_dis.shape[0] == k and cv_dis.shape[1] == k
# cv_dis += torch.eye(400)*1000
torch.save(cv_dis, os.path.join(os.path.dirname(args.content),
                                f'{args.model_name}_cv_dis_{k}x{k}x{n_samples_remain}.pth'))

cv_dis = cv_dis.mean(-1)

# kmeans
for nb in tqdm.tqdm(args.basis_number):
    kmeans = KMeans(n_clusters=nb, random_state=0).fit(cv_dis)
    centers = kmeans.cluster_centers_ # [10, 400]
    dis_mat_l1 = np.abs(centers[:,None,:] - cv_dis.numpy()[None, :, :]).mean(-1) # [10,400]
    print(np.min(dis_mat_l1, axis=-1), np.argmin(dis_mat_l1, axis=-1))
    if not os.path.exists('basis'): os.mkdir('basis')
    # np.save(f'basis/{args.model_name}_basis_{k}_id_{nb}.npy', np.array(sorted(np.argmin(dis_mat_l1, axis=-1))))
    np.savetxt(f'basis/{args.model_name}_basis_{k}_id_{nb}.txt', np.array(sorted(np.argmin(dis_mat_l1, axis=-1))), newline=' ',fmt='%d')
    
# vis
# imgs = []
# for i in np.array(sorted(np.argmin(dis_mat_l1, axis=-1))):
#     img = cv2.imread('data/data_221_S128F80_Base50_2x2_format/{:04}.png'.format(i))
#     imgs.append(img)

# plt.figure(dpi=200)
# plt.imshow(np.hstack(imgs))
