import cv2
# import matplotlib.pyplot as plt
import glob
import torch
import numpy as np
from tqdm import tqdm

imgs_400 = [cv2.imread(fn) for fn in sorted(glob.glob('./data/data_400_hard_50_2x2_format/*png'))]

# sv = torch.load('embedding_net400_d400/style.pth')

# sv_dis = (sv[None,...] - sv[:,None,:]).abs().mean(-1) # 400x400
# sv_dis += torch.eye(400)*1000 # kill eye

# score, sv_idx = torch.min(sv_dis, 1)

# out_imgs = []
# for i in range(400):
#     img_src = imgs_400[i]
#     img_nn = imgs_400[sv_idx[i]]
#     img_cat = np.hstack([img_src, img_nn])
#     img_cat[-2:] = img_cat[:2] = img_cat[:,-2:] = img_cat[:,:2] = 0
#     out_imgs.append(img_cat)

# # 8x50
# out_imgs = np.array(out_imgs)
# out_img = np.vstack([np.hstack(out_imgs[i*8: (i+1)*8]) for i in range(50)])
# cv2.imwrite('out.png', out_img)

### 
cv = torch.load('embedding_net400_d400/c_src.pth')

cv = cv.reshape([400,-1])

# cv_dis = (cv[None,...] - cv[:,None,:]).abs().mean(-1) # 400x400
# cv_dis += torch.eye(400)*1000 # kill eye

# avoid OOM
cv_dis_s = []
per = 5
for i in tqdm(range(400//per)):
    cv_dis = (cv[:,None,:] - cv[i*per:(i+1)*per][None,...]).abs().mean(-1) # [400, 1, k] - [1, 20,k] -> [400, 20, k] -> [400,20]
    cv_dis_s.append(cv_dis)

cv_dis = torch.cat(cv_dis_s, 1)
assert cv_dis.shape[0] == 400 and cv_dis.shape[1] == 400
cv_dis += torch.eye(400)*1000

score, cv_idx = torch.min(cv_dis, 1)

out_imgs = []
for i in range(400):
    img_src = imgs_400[i]
    img_nn = imgs_400[cv_idx[i]]
    img_cat = np.hstack([img_src, img_nn])
    img_cat[-2:] = img_cat[:2] = img_cat[:,-2:] = img_cat[:,:2] = 0
    out_imgs.append(img_cat)

# 8x50
out_imgs = np.array(out_imgs)
out_img = np.vstack([np.hstack(out_imgs[i*8: (i+1)*8]) for i in range(50)])
cv2.imwrite('out_c.png', out_img)