import argparse
from collections import OrderedDict
from genericpath import exists
from tqdm import tqdm, trange
import time
import numpy as np

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
# import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.utils as vutils

from models.generator import Generator as Generator
from models.discriminator import Discriminator as Discriminator
from models.guidingNet import GuidingNet

from train.train import trainGAN
from validation.validation import validateUN

from tools.utils import *
from datasets.datasetgetter import get_dataset
from tools.ops import calc_recon_loss, calc_abl
# from oss_client import OSSCTD

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0,type=int)
parser.add_argument("--save_path", default='./', help="where to store images")
parser.add_argument('--data_path', type=str, default='../data',
                    help='Dataset directory. Please refer Dataset in README.md')
parser.add_argument('--workers', default=4, type=int, help='the number of workers of data loader')
parser.add_argument('--font_len', default=5, type=int, help='the font id for style reference [style]')

parser.add_argument('--sty_dim', default=128, type=int, help='The size of style vector')
parser.add_argument('--sty_batch_size', default=40, type=int, help='batch size to calc style vector')
parser.add_argument('--output_k', default=400, type=int, help='Total number of classes to use')
parser.add_argument('--img_size', default=80, type=int, help='Input image size')

parser.add_argument('--load_model', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('--baseline_idx', default=2, type=int, help='the index of baseline. \
    0: the old baseline. 1: baseline that move the place of DCN. 2: Add addtional ADAIN_Conv based 1. 3: Delete last ADAIN based 1')
parser.add_argument('--load_style', type=str, default='', help='load style')

# FT
parser.add_argument('--ft_epoch', default=0, type=int, help='the number of epochs for style vector finetune')
parser.add_argument('--w_rec', default=0.1, type=float, help='Coefficient of Rec. loss of G')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--abl', action='store_true', help='using ABL')

args = parser.parse_args()
args.val_num = 40
args.val_bs = 30

cudnn.benchmark = False


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    args.device = torch.device('cuda')
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    global local_rank
    local_rank = args.rank
    print('| distributed init (rank {}-{})'.format(
    args.rank, local_rank), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend,
            init_method='env://',
                                     world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def main():
    st_main = time.time()
    args.num_cls = args.output_k
    args.data_dir = args.data_path

    args.att_to_use = list(range(args.font_len + 1))

    init_distributed_mode(args)

    args.epoch_acc = []
    args.epoch_avg_subhead_acc = []
    args.epoch_stats = []

    networks, opts = build_model(args)
    load_model(args, networks, opts)
    dataset, _ = get_dataset(args, with_path=True)
    inf(dataset, networks, opts, 999, args)
    dist.barrier()
    print('Using ', time.time() - st_main)

def inf(dataset, networks, opts, epoch, args):
    print(f'>>>>{local_rank}-{args.rank}')
    # set nets
    if not args.rank and not os.path.exists(args.save_path): os.mkdir(args.save_path)
    base_folders = [os.path.join(args.save_path, f'id_{i}') for i in range(args.font_len)]
    for base_folder in base_folders:
        if not args.rank and not os.path.exists(base_folder): os.mkdir(base_folder)
    source_id = args.font_len #0
    
    # ref_num = 10
    ref_idxs = list(range(args.font_len))
    dataset_val = None
    val_dataset = dataset['FULL']

    # load all data
    x_each_cls = []

    with torch.no_grad():
        val_tot_tars = torch.tensor(val_dataset.targets)

        style_cls_set = (val_tot_tars != args.att_to_use[args.font_len]).nonzero(as_tuple=False) # last [0 - k-1]
        style_val_dataset = torch.utils.data.Subset(val_dataset, style_cls_set)
        style_sampler = torch.utils.data.distributed.DistributedSampler(style_val_dataset)
        assert not len(style_cls_set) % (args.world_size * args.sty_batch_size), f'for ddp, {len(style_cls_set)} {args.world_size} {args.sty_batch_size}'
        style_dl = torch.utils.data.DataLoader(style_val_dataset, batch_size=args.sty_batch_size, 
                sampler=style_sampler, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        # [sample, path, idx]
        print(len(style_dl))

        G = networks['G'] 
        C = networks['C'] 
        C_EMA = networks['C_EMA'] 
        G_EMA = networks['G_EMA'] 
        G.eval()
        C.eval()
        C_EMA.eval()
        G_EMA.eval()
    
        # collect style vectors
        s_refs = torch.zeros(args.font_len, 128).to(args.device)
        s_refs_count = torch.zeros(args.font_len).to(args.device)
        if args.load_style == '':
            with torch.no_grad():
                for samples, paths, idxs in tqdm(style_dl) if local_rank==0 else style_dl:
                    x_ref = samples.to(args.device) # all ref
                    idxs = idxs.to(args.device)
                    s_ref = C_EMA(x_ref, sty=True)
                    s_refs.index_add_(0, idxs, s_ref) # [k, 128]
                    s_refs_count.index_add_(0, idxs, torch.ones_like(idxs) * 1.) # [k, 128]
                dist.barrier()
                dist.all_reduce(s_refs, op=dist.ReduceOp.SUM)
                dist.all_reduce(s_refs_count, op=dist.ReduceOp.SUM)
                dist.barrier()
                print(s_refs_count[0])
                assert torch.all(s_refs_count == s_refs_count[0])
                s_refs = s_refs / s_refs_count[:, None]
        else:
            s_refs = torch.load(args.load_style).to(args.device)

    # args.ft_epoch = 10
    if local_rank == 0: torch.save(s_refs, os.path.join(args.save_path, 'style_vec.pth'%(s_refs)))
    if args.ft_epoch > 0:
        s_refs = ft_style_all(s_refs, style_val_dataset, G_EMA, args)

    # Run inf
    content_cls_set = (val_tot_tars == args.att_to_use[args.font_len]).nonzero(as_tuple=False) # last [k]
    content_val_dataset = torch.utils.data.Subset(val_dataset, content_cls_set)
    content_sampler = torch.utils.data.distributed.DistributedSampler(content_val_dataset)
    content_dl = torch.utils.data.DataLoader(content_val_dataset, batch_size=30, sampler=content_sampler, shuffle=False,
                                             num_workers=4, pin_memory=True, drop_last=False)
    print(len(content_dl))
    with torch.no_grad():
        for samples, paths, _ in tqdm(content_dl) if local_rank==0 else content_dl:
            samples = samples.to(args.device)
            c_src, skip1, skip2 = G_EMA.module.cnt_encoder(samples)
            for s_id, s_ref in enumerate(tqdm(s_refs, leave=False)) if local_rank == 0 else enumerate(s_refs):
                s_ref = s_ref[None, :].repeat((samples.shape[0], 1))
                x_res_ema, _ = G_EMA.module.decode(c_src, s_ref, skip1, skip2)

                for img, path in zip(x_res_ema, paths):
                    base_folder = os.path.join(args.save_path, f'id_{s_id}')
                    vutils.save_image(img, os.path.join(base_folder, os.path.basename(path)), normalize=True, nrow=1)

#################
# Sub functions #
#################
def print_args(args):
    for arg in vars(args):
        print('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))

def ft_style_all(s_refs, dataset_ft, G_EMA, args):
    # fineturning style ref vec
    ep=-1
    s_refs = Variable(s_refs.detach(), requires_grad=True)
    opt = torch.optim.Adam([s_refs], args.lr, weight_decay=0.001)
    opt_G = torch.optim.Adam(G_EMA.parameters(), args.lr, weight_decay=0.001)
    
    style_sampler = torch.utils.data.distributed.DistributedSampler(dataset_ft)
    style_dl = torch.utils.data.DataLoader(dataset_ft, batch_size=20, sampler=style_sampler, shuffle=False,
                                        num_workers=8, pin_memory=True, drop_last=False) # not ddp
    
    ep_bar = tqdm(range(args.ft_epoch)) if local_rank == 0 else range(args.ft_epoch)
    save_suffix=''
    if args.abl: save_suffix += '_abl'
    for ep in ep_bar:
        if local_rank == 0:  ep_bar.set_description(f"Epochs")
        for step, (imgs_gt, _, idxs) in enumerate(tqdm(style_dl, leave=False)) if local_rank == 0 else enumerate(style_dl):
            imgs_in = imgs_gt
            s_ref_batch = s_refs[idxs] #s_ref.repeat((imgs_gt.shape[0], 1))
            
            if local_rank == 0:
                print(idxs)
                vutils.save_image(imgs_gt, os.path.join('vvv.png'), normalize=True,
                        nrow=4)
            exit()
            x_org = imgs_in.to(args.device)
            x_gt = imgs_gt.to(args.device)

            c_src, skip1, skip2 = G_EMA.module.cnt_encoder(x_org)
            x_rec, _ = G_EMA.module.decode(c_src, s_ref_batch, skip1, skip2)
            g_imgrec = calc_recon_loss(x_rec, x_gt)

            g_loss = args.w_rec * g_imgrec

            info = dict()
            info['rec_loss'] = args.w_rec * g_imgrec

            # abl
            if args.abl:
                g_img_abl = calc_abl(x_rec, x_gt)
                if g_img_abl is not None:
                    g_loss += args.w_rec * g_img_abl
                    info['abl'] = args.w_rec * g_img_abl
            

            if local_rank == 0:  ep_bar.set_postfix(info)
            s_refs.retain_grad()
            opt.zero_grad()
            opt_G.zero_grad()
            g_loss.backward()
            opt.step()
        if local_rank == 0: torch.save(s_refs, os.path.join(args.save_path,
            'style_vec_ft{}_ep{}.pth'.format(save_suffix, ep+1)))

    return s_refs

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
        net_tmp = net.to(args.device)
        #net_tmp = net.to(device)
        # networks[name] = net_tmp
        networks[name] =  torch.nn.parallel.DistributedDataParallel(net_tmp, device_ids=[args.gpu])

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
                    for key, val in checkpoint[name + '_state_dict'].items(): # TODO: Format
                        # tmp_new_dict[key[7:]] = val
                        tmp_new_dict[key] = val
                    net.load_state_dict(tmp_new_dict, strict=True)
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
