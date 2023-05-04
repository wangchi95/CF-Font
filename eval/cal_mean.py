import argparse
import os
import torch
import numpy as np
import time
import pdb

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f','--folder', type=str, default='../test/tmp_base')
parser.add_argument('-j','--jump', type=int, default=[], nargs='+')
parser.add_argument('-k','--k', type=int, default=240)
parser.add_argument('-c','--choice', type=str, default='fs')
args = parser.parse_args()

def get_kv(line, sp=None):
    if line[-1] == '\n': line = line[:-1]
    try:
        k, v = line.split(sp)
        v = float(v.lstrip())
        return k, v
    except:
        return None, None

scores = {}

def put_in_dict(k, v, d):
    if k is None: return
    if k in d.keys():
        d[k].append(v)
    else:
        d[k] = [v]

for i in range(args.k):
    if i in args.jump: continue
    f_fn = os.path.join(args.folder, f'id_{i}_fid.txt')
    s_fn = os.path.join(args.folder, f'id_{i}_scores.txt')
    # fid
    if 'f' in args.choice:
        with open(f_fn, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0: print(i)
        for line in lines:
            k, v = get_kv(line, sp=':')
            put_in_dict(k,v,scores)
    # scores
    if 's' in args.choice:
        with open(s_fn, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'Detail' in line: break
            k, v = get_kv(line)
            put_in_dict(k,v,scores)

for k, v in scores.items():
    print(f'[{len(v)}]', k, np.mean(v))

for k, d in [['l1', 5], ['rmse', 4], ['ssim', 4], ['lpips', 5], ['FID', 4]]:
    if k in scores.keys():
        #print('{}'.format(np.mean(scores[k]).round(d)), end=' ')
        #print(f'%.{d}f'%(np.mean(scores[k])), end=' ')
        print(f'%.{d}f'%(np.mean(scores[k])))
print('')
