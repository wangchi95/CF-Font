import argparse
from collections import OrderedDict

import numpy as np

import time
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
# import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from models.generator import Generator as Generator
from models.discriminator import Discriminator as Discriminator
from models.guidingNet import GuidingNet
from tools.utils import *
from datasets.datasetgetter import get_dataset
from tqdm import tqdm, trange

# from oss_client import OSSCTD

# Configuration
parser = argparse.ArgumentParser(description='PyTorch GAN Training')
parser.add_argument('--data_path', type=str, default='data/imgs/Seen240_S80F50_TRAIN800',
                    help='Dataset directory. Please refer Dataset in README.md')
parser.add_argument('--basis_path', type=str, default='data/imgs/BASIS_S80F50_TRAIN800',
                    help='Basis directory. Please refer Dataset in README.md')
parser.add_argument('--workers', default=4, type=int, help='the number of workers of data loader')
parser.add_argument('--font_len', default=5, type=int, help='the font id for style reference [style]')
parser.add_argument('--basis_len', default=5, type=int, help='the font id for basis')

parser.add_argument('--sty_dim', default=128, type=int, help='The size of style vector')
parser.add_argument('--output_k', default=400, type=int, help='Total number of classes to use')
parser.add_argument('--img_size', default=80, type=int, help='Input image size')

parser.add_argument('--load_model', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('--baseline_idx', default=2, type=int, help='the index of baseline. \
    0: the old baseline. 1: baseline that move the place of DCN. 2: Add addtional ADAIN_Conv based 1. 3: Delete last ADAIN based 1')
parser.add_argument('--load_style', type=str, default='', help='load style')

parser.add_argument('-t', '--temperature', type=float, default=0.01, help='softmax temperature')

parser.add_argument('--zero_eye', action='store_true', help='set eye to zero')
parser.add_argument('--save_fn', type=str, default='', help='save weight name')

args = parser.parse_args()
args.bs_per_font = 40
args.mini_batch = args.val_num = 6
args.local_rank = 0

def main():
    st_main = time.time()
    args.data_dir = args.data_path

    args.att_to_use = list(range(args.font_len))
    args.att_to_use_basis = list(range(args.basis_len))

    args.epoch_acc = []
    args.epoch_avg_subhead_acc = []
    args.epoch_stats = []

    networks, opts = build_model(args)
    load_model(args, networks, opts)
    dataset, _ = get_dataset(args)
    print(args.basis_path)
    basis_dataset, _ = get_dataset(args, data_dir=args.basis_path, class_to_use=args.att_to_use_basis)
    inf(networks, dataset['FULL'], basis_dataset['FULL'], args)
    print('Using ', time.time() - st_main)

def load_to_list(ds, att):
    # import pdb; pdb.set_trace()
    # load all data
    each_cls = []
    with torch.no_grad():
        val_tot_tars = torch.tensor(ds.targets)
        with trange(len(att)) as t:
            for cls_idx in t:
                t.set_description('Loading Data')
                tmp_cls_set = (val_tot_tars == att[cls_idx]).nonzero()
                tmp_ds = torch.utils.data.Subset(ds, tmp_cls_set)
                tmp_dl = torch.utils.data.DataLoader(tmp_ds, batch_size=args.bs_per_font, shuffle=False,
                                                    num_workers=4, pin_memory=True, drop_last=False)
                cls_now = torch.cat([x.clone() for x, _ in tmp_dl], 0)
                each_cls.append(cls_now)
                del tmp_dl
    return each_cls


def inf(networks, dataset, basis_dataset, args):
    # set nets
    basis_each_cls = load_to_list(basis_dataset, args.att_to_use_basis)
    x_each_cls = load_to_list(dataset, args.att_to_use)
    chars_num = len(dataset) // args.font_len

    G_EMA = networks['G_EMA'] 
    G_EMA.eval()
    
    refs_bar = trange(args.font_len)
    st = time.time()
    tot_idx = 0

    ws = [] # 400, basis

    for i in range(len(x_each_cls)):
        assert len(x_each_cls[i]) == chars_num
    for i in range(len(basis_each_cls)):
        assert len(basis_each_cls[i]) == chars_num
    
    basis_each_cls = torch.stack(basis_each_cls) # [10, 404, 3, h, w]

    with torch.no_grad():
        for s_id_now in refs_bar: # [1, basis]
            refs_bar.set_description(f"Ref")

            ws_i = []
            
            # if chars_num % args.mini_batch != 0:
            #     print('cannot be divided without a remainder, set mini_batch to 1')
            #     args.mini_batch = 1
            for idx, (cnt_idx) in enumerate(trange((int)(np.ceil(chars_num/args.mini_batch)), leave=False)):
                idx_min_now = cnt_idx * args.mini_batch
                idx_max_now = min(chars_num, (cnt_idx+1) * args.mini_batch)

                x_src = x_each_cls[s_id_now][idx_min_now:idx_max_now, :, :, :].cuda(non_blocking=True)
                c_src, skip1, skip2 = G_EMA.cnt_encoder(x_src)
                x_basis_now = basis_each_cls[:, idx_min_now:idx_max_now].cuda(non_blocking=True)  # [10, val_batch_now, 3, h, w]
                basis_shape = x_basis_now.shape
                #c_src_basis, _, _ = G_EMA.cnt_encoder(x_basis_now.reshape(-1, *basis_shape[2:]))
                #c_src_basis = c_src_basis.reshape(*basis_shape[:2], *c_src_basis.shape[1:])
                c_src_basis_list = [G_EMA.cnt_encoder(x_basis_now[bi])[0] for bi in range(basis_shape[0])]
                c_src_basis = torch.stack(c_src_basis_list)
                weight_now = get_logits_dis(c_src[None,...], c_src_basis) # [10]
                ws_i.append(weight_now)

                info = dict()
                info['fps'] = tot_idx/(time.time()-st)
                refs_bar.set_postfix(info)
            logits_i = torch.mean(torch.stack(ws_i),0)/args.temperature
            if args.zero_eye:
                logits_i[s_id_now] = -100 # mask eye
            ws_prob = torch.nn.functional.softmax(logits_i)
            ws.append(ws_prob)
        ws = torch.stack(ws).cpu()
        torch.save(ws, args.save_fn)
#################
# Sub functions #
#################

def get_logits_dis(tgt_feature, base_feature,opt='l1'):
    assert opt in ['inner', 'l1', 'l2']
    base_n = base_feature.shape[0]
    if opt == 'inner':   
        w_inner = tgt_feature * base_feature # [10,40,262144]
        w_mean = torch.mean(w_inner.reshape(base_n,-1), 1)
    elif opt == 'l1':
        w_l1 = tgt_feature - base_feature # [10,40,262144]
        w_mean = -torch.mean(w_l1.abs().reshape(base_n,-1), 1)
    elif opt == 'l2':
        w_l2 = (tgt_feature - base_feature)**2 # [10,40,262144]
        w_mean = -torch.mean(w_l2.reshape(base_n,-1), 1)
    return w_mean

def print_args(args):
    for arg in vars(args):
        print('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))


def build_model(args):
    args.to_train = 'CDG'

    networks = {}
    opts = {}
    if 'C' in args.to_train:
        networks['C'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
        networks['C_EMA'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
    if 'D' in args.to_train:
        networks['D'] = Discriminator(args.img_size, num_domains=args.output_k)
    if 'G' in args.to_train:
        networks['G'] = Generator(args.img_size, args.sty_dim, use_sn=False, mute=True, baseline_idx=args.baseline_idx)
        networks['G_EMA'] = Generator(args.img_size, args.sty_dim, use_sn=False, mute=True, baseline_idx=args.baseline_idx)

    for name, net in networks.items():
        net_tmp = net.cuda()
        networks[name] = net_tmp #torch.nn.parallel.DistributedDataParallel(net_tmp ,device_ids=[local_rank],
                                                    # output_device=local_rank)

    if 'C' in args.to_train:
        opts['C'] = torch.optim.Adam(networks['C'].parameters(), 1e-4, weight_decay=0.001)
        networks['C_EMA'].load_state_dict(networks['C'].state_dict())
    if 'D' in args.to_train:
        opts['D'] = torch.optim.RMSprop(networks['D'].parameters(), 1e-4, weight_decay=0.0001)
    if 'G' in args.to_train:
        opts['G'] = torch.optim.RMSprop(networks['G'].parameters(), 1e-4, weight_decay=0.0001)

    return networks, opts


def load_model(args, networks, opts):
    if args.load_model is not None:
        load_file = args.load_model
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']

            for name, net in networks.items():
                tmp_keys = next(iter(checkpoint[name + '_state_dict'].keys()))
                if 'module' in tmp_keys:
                    tmp_new_dict = OrderedDict()
                    for key, val in checkpoint[name + '_state_dict'].items():
                        tmp_new_dict[key[7:]] = val
                        # tmp_new_dict[key] = val
                    net.load_state_dict(tmp_new_dict, strict=False)
                    networks[name] = net
                else:
                    net.load_state_dict(checkpoint[name + '_state_dict'])
                    networks[name] = net

            for name, opt in opts.items():
                opt.load_state_dict(checkpoint[name.lower() + '_optimizer'])
                opts[name] = opt
            print("=> loaded checkpoint '{}' (epoch {})".format(load_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.load_model))

if __name__ == '__main__':
    main()
