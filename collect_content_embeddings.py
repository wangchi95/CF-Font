import argparse
from genericpath import exists
import warnings
from datetime import datetime
from glob import glob
from shutil import copyfile
from collections import OrderedDict

import numpy as np

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
from models.inception import InceptionV3

from train.train import trainGAN

from validation.validation import validateUN

from tools.utils import *
from datasets.datasetgetter import get_dataset
from tools.ops import initialize_queue

import torchvision.utils as vutils
from tqdm import tqdm

import pdb

from tools.ops import compute_grad_gp, update_average, copy_norm_params, queue_data, dequeue_data, \
    average_gradients, calc_adv_loss, calc_contrastive_loss, calc_recon_loss, calc_abl

# from oss_client import OSSCTD

# Configuration
parser = argparse.ArgumentParser(description='PyTorch GAN Training')
parser.add_argument("--save_path", default='../vis', help="where to store images")

parser.add_argument('--data_path', type=str, default='../data',
                    help='Dataset directory. Please refer Dataset in README.md')
parser.add_argument('--workers', default=4, type=int, help='the number of workers of data loader')

parser.add_argument('--model_name', type=str, default='GAN',
                    help='Prefix of logs and results folders. '
                         'ex) --model_name=ABC generates ABC_20191230-131145 in logs and results')

parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--val_batch', default=1, type=int, help='Batch size for validation. '
                         'The result images are stored in the form of (val_batch, val_batch) grid.')
parser.add_argument('--ref_num', default=10, type=int, help='Number of images as reference')
parser.add_argument('--s_id', default=5, type=int, help='the font id for style reference [style]')
parser.add_argument('--c_id', default=0, type=int, help='the font id for content [content]')
parser.add_argument('--ft_id', default=0, type=int, help='the font id for finetune [finetune]')

parser.add_argument('--sty_dim', default=128, type=int, help='The size of style vector')
parser.add_argument('--output_k', default=400, type=int, help='Total number of classes to use')
parser.add_argument('--img_size', default=80, type=int, help='Input image size')
parser.add_argument('--dims', default=2048, type=int, help='Inception dims for FID')

parser.add_argument('--ft_epoch', default=0, type=int, help='the number of epochs for style vector finetune')

parser.add_argument('--w_rec', default=0.1, type=float, help='Coefficient of Rec. loss of G')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

parser.add_argument('--load_style', type=str, default='', help='load style')

parser.add_argument('--abl', action='store_true', help='using ABL')
parser.add_argument('--no_skip', action='store_true', help='not save skip')
# parser.add_argument('--vis', action='store_true', help='vis result')
parser.add_argument('--vis', type=str, default='', help='vis result path')

parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)'
                         'ex) --load_model GAN_20190101_101010'
                         'It loads the latest .ckpt file specified in checkpoint.txt in GAN_20190101_101010')
parser.add_argument('--n_atts', default=400, type=int, help='The size of atention maps')

parser.add_argument('--baseline_idx', default=2, type=int, help='the index of baseline. \
    0: the old baseline. 1: baseline that move the place of DCN. 2: Add addtional ADAIN_Conv based 1. 3: Delete last ADAIN based 1')

parser.add_argument('--nocontent', action='store_true', help='no content')


args = parser.parse_args()
args.val_num = 30
args.local_rank = 0

n_atts = args.n_atts
def main():
    args.num_cls = args.output_k
    args.data_dir = args.data_path

    args.att_to_use = list(range(n_atts))

    # IIC statistics
    args.epoch_acc = []
    args.epoch_avg_subhead_acc = []
    args.epoch_stats = []

    # build model - return dict
    networks, opts = build_model(args)

    # load model if args.load_model is specified
    load_model(args, networks, opts)

    # All the test is done in the training - do not need to call
    dataset, _ = get_dataset(args)
    inf(dataset, networks, opts, 999, args)


def inf(dataset, networks, opts, epoch, args):
    # set nets
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    # data loader
    val_dataset = dataset['FULL']

    # load all data    
    C_EMA = networks['C_EMA'] 
    G_EMA = networks['G_EMA'] 
    C_EMA.eval()
    G_EMA.eval()

    with torch.no_grad():
        val_tot_tars = torch.tensor(val_dataset.targets)
        s_refs = []
        c_srcs = []
        skip1s = []
        skip2s = []
        for cls_idx in tqdm(range(n_atts)):
            tmp_cls_set = (val_tot_tars == cls_idx).nonzero()
            val_num = len(tmp_cls_set)
            tmp_ds = torch.utils.data.Subset(val_dataset, tmp_cls_set)
            tmp_dl = torch.utils.data.DataLoader(tmp_ds, batch_size=val_num,
                    shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
            tmp_iter = iter(tmp_dl)
            tmp_sample = None
            for sample_idx in range(len(tmp_iter)):
                imgs, _ = next(tmp_iter)
                x_ = imgs
                tmp_sample = x_.clone() if tmp_sample is None else torch.cat((tmp_sample, x_), 0)
                
            x_ref = tmp_sample.cuda() # all ref
            s_ref = C_EMA(x_ref, sty=True)
            s_refs.append(torch.mean(s_ref.detach().cpu(), dim=0, keepdim=True)) # average

            if not args.nocontent:
                c_src, skip1, skip2 = G_EMA.cnt_encoder(x_ref)
                # pdb.set_trace()
                c_srcs.append(c_src.detach().cpu().unsqueeze(0))
                if not args.no_skip:
                    skip1s.append(skip1.detach().cpu().unsqueeze(0))
                    skip2s.append(skip2.detach().cpu().unsqueeze(0))
        


        s_ref = torch.cat(s_refs, dim=0)
        ref_fn = os.path.join(args.save_path, 'style.pth')
        torch.save(s_ref, ref_fn)
        if not args.nocontent:
            c_src = torch.cat(c_srcs, dim=0)
            c_src_fn = os.path.join(args.save_path, 'c_src.pth')
            torch.save(c_src, c_src_fn)
            if not args.no_skip:
                skip1 = torch.cat(skip1s, dim=0)
                skip2 = torch.cat(skip2s, dim=0)
                skip1_fn = os.path.join(args.save_path, 'skip1.pth')
                skip2_fn = os.path.join(args.save_path, 'skip2.pth')
                torch.save(skip1, skip1_fn)
                torch.save(skip2, skip2_fn)


    

#################
# Sub functions #
#################
def print_args(args):
    for arg in vars(args):
        print('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))


def build_model(args):
    args.to_train = 'CG'

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
