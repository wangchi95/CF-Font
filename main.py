import argparse
from collections import OrderedDict
from datetime import datetime
from genericpath import exists
import warnings
from glob import glob
from shutil import copyfile
import os

import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
# import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from models.generator import Generator
from models.discriminator import Discriminator
from models.guidingNet import GuidingNet
from models.inception import InceptionV3
import numpy as np

from train.train import trainGAN
from train.train_cf import trainGANCF
from validation.validation import validateUN

from datasets.datasetgetter import get_dataset, get_cf_dataset
from tools.utils import makedirs, save_checkpoint
from tools.ops import initialize_queue
try:
    from oss_client import OSSCTD
except:
    print('Warning! oss_client is not loaded.')


# ------------------------------ configuration ----------------------------------
parser = argparse.ArgumentParser(description='PyTorch GAN Training')
parser.add_argument("--local_rank", type=int, default=0, help="gpu number")
parser.add_argument('--on_oss', dest='on_oss', action='store_true', help='using oss data')
parser.add_argument('--remote', dest='remote', action='store_true', help='run on pai')
parser.add_argument("--save_path", default='./', help="where to store images")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

parser.add_argument('--data_path', type=str, default='../data',
                    help='Dataset directory. Please refer Dataset in README.md')
parser.add_argument('--workers', default=4, type=int, help='the number of workers of data loader')

parser.add_argument('--model_name', type=str, default='GAN',
                    help='Prefix of logs and results folders. '
                         'ex) --model_name=ABC generates ABC_20191230-131145 in logs and results')

parser.add_argument('--epochs', default=250, type=int, help='Total number of epochs to run. Not actual epoch.')
parser.add_argument('--iters', default=1000, type=int, help='Total number of iterations per epoch')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--val_num', default=190, type=int,help='Number of test images for each style')
parser.add_argument('--val_batch', default=10, type=int,
                    help='Batch size for validation. '
                         'The result images are stored in the form of (val_batch, val_batch) grid.')
parser.add_argument('--log_step', default=100, type=int)

parser.add_argument('--sty_dim', default=128, type=int, help='The size of style vector')
parser.add_argument('--output_k', default=400, type=int, help='Total number of classes to use')
parser.add_argument('--img_size', default=80, type=int, help='Input image size')
parser.add_argument('--dims', default=2048, type=int, help='Inception dims for FID')

parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)'
                         'ex) --load_model GAN_20190101_101010'
                         'It loads the latest .ckpt file specified in checkpoint.txt in GAN_20190101_101010')
parser.add_argument('--validation', dest='validation', action='store_true', help='Call for valiation only mode')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use.')
parser.add_argument('--ddp', dest='ddp', action='store_true', help='Call if using DDP')
parser.add_argument('--port', default='8993', type=str)

parser.add_argument('--iid_mode', default='iid+', type=str, choices=['iid', 'iid+'])

parser.add_argument('--w_gp', default=10.0, type=float, help='Coefficient of GP of D')
parser.add_argument('--w_hsic', default=1.0, type=float, help='Coefficient of hsic. loss of G')
parser.add_argument('--w_grec', default=0.1, type=float, help='Coefficient of Global Rec. loss of G')
parser.add_argument('--w_rec', default=0.1, type=float, help='Coefficient of Rec. loss of G')
parser.add_argument('--w_adv', default=1.0, type=float, help='Coefficient of Adv. loss of G')
parser.add_argument('--w_vec', default=0.01, type=float, help='Coefficient of Style vector rec. loss of G')
parser.add_argument('--w_off', default=0.5, type=float, help='Coefficient of offset normalization. loss of G')
parser.add_argument('--w_wdl', default=0.01, type=float, help='Coefficient of wdl of G(over L1)')
parser.add_argument('--w_pkl', default=0.05, type=float, help='Coefficient of pkl of G(over L1)')

parser.add_argument('--style_con', dest='style_con', action='store_true', help='using style Rec. loss')
parser.add_argument('--recon_losses', action='store_true', help='using Global Rec. loss')
parser.add_argument('--abl', action='store_true', help='using abl')
parser.add_argument('--detach', action='store_true', help='detach gradient from x_fake')
parser.add_argument('--phl', action='store_true', help='using phl')
parser.add_argument('--wdl', action='store_true', help='using wdl')
parser.add_argument('--pkl', action='store_true', help='using pkl')
parser.add_argument('--no_l1', action='store_true', help='unuse l1 for g')
parser.add_argument('--quantize', action='store_true', help='quantize x_fake')

parser.add_argument('--baseline_idx', default=2, type=int, help='the index of baseline. \
    0: the old baseline. 1: baseline that move the place of DCN. 2: Add addtional ADAIN_Conv based 1. 3: Delete last ADAIN based 1')

parser.add_argument('-cf', '--content_fusion',action='store_true', help='use content fusion to train')
parser.add_argument('--base_idxs', default='', type=str)
parser.add_argument('--base_ws', default='', type=str)
parser.add_argument('--load_model_oss', default='', type=str)
parser.add_argument('-btn', '--base_top_n', default=-1, type=int, help='seletc only top n basis. -1 for all')
parser.add_argument('--no_val', action='store_true', help='skip all val')

args = parser.parse_args()


oss_client = OSSCTD() if args.on_oss else None
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


# ------------------------------ environment ----------------------------------
info = ''
info += 'Cuda is available: {} | '.format(torch.cuda.is_available())
info += 'Cuda count: {} | '.format(torch.cuda.device_count())
info += 'CUDA: {} | '.format(torch.version.cuda)
info += 'torch: {} | '.format(torch.__version__)

local_rank = 0
args.distributed = False
if args.ddp or args.remote:
    args.distributed = True
    cudnn.benchmark = True
    dist.init_process_group(init_method='env://', backend="nccl")

    local_rank = int(dist.get_rank())
    world_size = dist.get_world_size()

    info += 'Wordsize: {} | '.format(world_size)
    info += 'Rank: {} | '.format(local_rank)

print(info)

# -----------------------------------------------------------------------------


def main():
    ####################
    # Default settings #
    ####################
    assert args.on_oss or os.path.exists(args.save_path)
    assert args.val_batch <= args.val_num or (not args.on_oss)

    args.local_rank = local_rank
    if not args.remote:
        torch.cuda.set_device(local_rank) # ! TODO: remove

    args.train_mode = 'GAN'
    args.start_epoch = 0
    args.data_dir = args.data_path
    
    args.save_path_oss = ''
    if args.on_oss:
        assert args.save_path.startswith('xxx/DG-FONT')
        args.save_path_oss = args.save_path
        args.save_path = '/tmp/save_exp' if args.remote else './'
        makedirs(args.save_path)

    args.unsup_start = 0 # unsup_start : train networks with supervised data only before unsup_start
    args.separated = 0 # separated : train IIC only until epoch = args.separated
    args.ema_start = 1 # ema_start : Apply EMA to Generator after args.ema_start
    args.fid_start = 0

    if args.content_fusion:
        # args.base_idxs = np.load(args.base_idxs)
        args.base_idxs = np.loadtxt(args.base_idxs, dtype='int')
        args.base_ws = torch.load(args.base_ws)

    if args.validation:
        args.val_src_reduce = 1
        args.val_ref_reduce = 1
    else:
        args.val_src_reduce = 1 # 400 // 20 = 20
        args.val_ref_reduce = 1 # 400 // 8 = 50

    # Logs / Results
    if args.load_model is None:
        args.model_name = '{}_{}'.format(args.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        args.model_name = os.path.basename(args.load_model)

    args.log_dir = os.path.join(args.save_path, 'logs', args.model_name)
    args.event_dir = os.path.join(args.log_dir, 'events')
    args.res_dir = os.path.join(args.save_path, 'results', args.model_name)
        
    args.log_dir_oss = os.path.join(args.save_path_oss, 'logs', args.model_name)
    args.event_dir_oss = os.path.join(args.log_dir_oss, 'events')
    args.res_dir_oss = os.path.join(args.save_path_oss, 'results', args.model_name)
    print(args)

    makedirs(args.log_dir)
    makedirs(os.path.join(args.save_path ,'logs'))
    makedirs(os.path.join(args.save_path , 'results'))
    makedirs(args.res_dir)
    
    if local_rank == 0:
        dirs_to_make = next(os.walk('./'))[1]
        not_dirs = ['.idea', '.git', 'logs', 'results', '.gitignore', '.nsmlignore', 'resrc']
        
        makedirs(os.path.join(args.log_dir, 'codes'))
        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            makedirs(os.path.join(args.log_dir, 'codes', to_make))

        copy_func = oss_client.write_file if args.on_oss else copyfile
        root_dir = args.log_dir_oss if args.on_oss else args.log_dir

        if args.load_model is None:
            pyfiles = glob("./*.py")
            for py in pyfiles:
                copy_func(py, os.path.join(root_dir, 'codes', py))

            for to_make in dirs_to_make:
                if to_make in not_dirs:
                    continue
                tmp_files = glob(os.path.join('./', to_make, "*.py"))
                for py in tmp_files:
                    copy_func(py, os.path.join(root_dir, 'codes', py[2:]))

    # #GT-classes
    args.num_cls = args.output_k #TODO: num_cls??

    # Classes to use
    if args.validation:
        args.att_to_use = [0, 1]
    else:
        args.att_to_use = list(range(args.output_k))

    # IIC statistics
    args.epoch_acc = []
    args.epoch_avg_subhead_acc = []
    args.epoch_stats = []

    # Logging
    if local_rank == 0:
        SUMMARY_DIR = os.environ['SUMMARY_DIR'] if 'SUMMARY_DIR' in os.environ.keys() else args.event_dir
        logger = SummaryWriter(args.event_dir) if not args.on_oss else SummaryWriter(log_dir=SUMMARY_DIR, max_queue=30, flush_secs=60)
        # summary_hook = xdl_runner.add_summary_hook()
        # summary_hook.summary('auc', auc, stype='scalar')
        # summary_hook.summary('loss', loss, stype='scalar')
        # sess = xdl.TrainSession(hooks=[ckpt_hook, log_hook, auc_hook,summary_hook])
    else:
        logger = None 
    
    # build model - return dict; load model if args.load_model is specified
    networks, opts = build_model(args)
    load_model(args, networks, opts)
    
    # get dataset and data loader
    train_dataset, val_dataset = get_dataset(args)
    if args.content_fusion:
        cf_dataset, cf_basis_dataset = get_cf_dataset(args)
        train_loader, val_loader, train_sampler, train_basis_loader, train_basis_sampler = get_loader_cf(
                args, {'train': train_dataset, 'val': val_dataset, 'cf':cf_dataset, 'cf_base': cf_basis_dataset})
    else:
        train_loader, val_loader, train_sampler = get_loader(args, {'train': train_dataset, 'val': val_dataset})

    # map the functions to execute - un / sup / semi-
    trainFunc, validationFunc = map_exec_func(args)

    # print all the argument
    if local_rank == 0:
        print_args(args)

    # All the test is done in the training - do not need to call
    if args.validation:
        validationFunc(val_loader, networks, 999, args, {'logger': logger}, oss_client=oss_client if args.on_oss else None)
        return

    # For saving the model
    if local_rank == 0:
        with open(os.path.join(args.log_dir, "record.txt"), "a+") as record_txt:
            for arg in vars(args):
                record_txt.write('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))
        if args.on_oss:
            oss_client.write_file(os.path.join(args.log_dir, "record.txt"), \
                os.path.join(args.log_dir_oss, "record.txt"))

    for epoch in range(args.start_epoch, args.epochs):
        save_every_10 = 3/4 * args.epochs
        save_every_1 = args.epochs - 10

        if local_rank == 0:
            t_ep_start = datetime.now()
            save_model(args, epoch, networks, opts)
            print("START EPOCH[{}] at: {}".format(epoch+1, t_ep_start.strftime("%Y.%m.%d-%H:%M:%S")))

        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch == args.ema_start and 'GAN' in args.train_mode:
            if args.distributed:
                networks['G_EMA'].module.load_state_dict(networks['G'].module.state_dict())
            else:
                networks['G_EMA'].load_state_dict(networks['G'].state_dict())
    
        if args.content_fusion:
            trainFunc(train_loader, train_basis_loader, networks, opts, epoch, args, {'logger': logger}, \
                    detach=args.detach, quantize=args.quantize, style_con=args.style_con, \
                    reconstruction_losses=args.recon_losses, abl=args.abl, \
                    oss_client=oss_client if args.on_oss else None)       
        else:
            trainFunc(train_loader, networks, opts, epoch, args, {'logger': logger}, \
                    detach=args.detach, quantize=args.quantize, style_con=args.style_con, \
                    reconstruction_losses=args.recon_losses, abl=args.abl, \
                    phl=args.phl, wdl=args.wdl)

        if local_rank == 0:
            print("TRAIN using: {}".format(datetime.now() - t_ep_start))
            t_ep_start = datetime.now()
        
        # Val & save models
        if not args.content_fusion and not args.no_val and \
            ((epoch + 1) % 20 == 0 or ((epoch + 1) % 10 == 0 and epoch >= save_every_10) or epoch >= save_every_1):
            print(len(val_loader))
            validationFunc(val_loader, networks, epoch, args, {'logger': logger}, oss_client=oss_client if args.on_oss else None)
            if 0 == local_rank:
                print("VAL using: {}".format(datetime.now() - t_ep_start))
                t_ep_start = datetime.now()


#################
# Sub functions #
#################
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
        networks['G'] = Generator(args.img_size, args.sty_dim, use_sn=False, mute=(local_rank==0), baseline_idx=args.baseline_idx)
        networks['G_EMA'] = Generator(args.img_size, args.sty_dim, use_sn=False, mute=(local_rank==0), baseline_idx=args.baseline_idx)

    for name, net in networks.items():
        net_tmp = net.cuda()
        if args.distributed:
            networks[name] = torch.nn.parallel.DistributedDataParallel(net_tmp, device_ids=[local_rank], output_device=local_rank)

    if 'C' in args.to_train:
        if args.distributed:
            opts['C'] = torch.optim.Adam(networks['C'].module.parameters(),
                    args.lr, weight_decay=0.001)
            networks['C_EMA'].module.load_state_dict(networks['C'].module.state_dict())
            
        else:
            opts['C'] = torch.optim.Adam(networks['C'].parameters(),
                    args.lr, weight_decay=0.001)
            networks['C_EMA'].load_state_dict(networks['C'].state_dict())
        
    if 'D' in args.to_train:
        opts['D'] = torch.optim.RMSprop(
            networks['D'].module.parameters() if args.distributed else networks['D'].parameters(),
            args.lr, weight_decay=0.0001)
    if 'G' in args.to_train:
        opts['G'] = torch.optim.RMSprop(
            networks['G'].module.parameters() if args.distributed else networks['G'].parameters(),
            args.lr, weight_decay=0.0001)

    return networks, opts


def load_model(args, networks, opts):
    if not args.on_oss and args.load_model_oss not in (None, ''):
        print(f'Warning! args.load_model_oss({args.load_model_oss}) is deleted.')
        args.load_model_oss = ''

    if args.load_model is None and args.load_model_oss in (None,''):
        print('Warning! No model to load.')
        return

    # get checkpoint path
    if args.on_oss and not os.path.exists(os.path.join(args.log_dir, "checkpoint.txt")):
        oss_client.fetch_file(os.path.join(args.log_dir_oss, "checkpoint.txt"), \
            os.path.join(args.log_dir, "checkpoint.txt"))

    check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
    to_restore = check_load.readlines()[-1].strip()
    to_restore_oss = to_restore.replace('model_','generator_').replace('.ckpt' ,'.pth')
    load_file = os.path.join(args.log_dir, to_restore)
    load_file_oss = load_file.replace('model_' ,'generator_').replace('.ckpt' ,'.pth')
    if not os.path.isfile(load_file) and os.path.isfile(load_file_oss):
        load_file = load_file_oss

    print(load_file_oss, os.path.isfile(load_file_oss))

    # load file
    if os.path.isfile(load_file):
        if local_rank == 0:
            print("=> loading checkpoint '{}'".format(load_file))
        if args.load_model_oss != '':
            oss_client.fetch_file(os.path.join(args.load_model_oss, "ckpt", to_restore_oss), load_file)
        checkpoint = torch.load(load_file, map_location='cpu')
        args.start_epoch = checkpoint['epoch']

        for name, net in networks.items():
            tmp_keys = next(iter(checkpoint[name + '_state_dict'].keys()))
            if 'module' in tmp_keys:
                tmp_new_dict = OrderedDict()
                for key, val in checkpoint[name + '_state_dict'].items():
                    # tmp_new_dict[key[7:]] = val
                    tmp_new_dict[key] = val
                net.load_state_dict(tmp_new_dict, strict=False)
            else:
                net.load_state_dict(checkpoint[name + '_state_dict'])
            networks[name] = net

        for name, opt in opts.items():
            opt.load_state_dict(checkpoint[name.lower() + '_optimizer'])
            opts[name] = opt
        if local_rank == 0:
            print(f"=> loaded checkpoint '{load_file}' (epoch {checkpoint['epoch']})")
    else:
        if local_rank == 0:
            print(f"=> no checkpoint found at '{args.log_dir}'")


def get_loader_cf(args, dataset, with_pair=False):
    train_dataset = dataset['cf']
    train_basis_dataset = dataset['cf_base']
    val_dataset = dataset['val']
    train_unpair_dataset = dataset['train']

    if local_rank == 0:
        print(len(val_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_basis_sampler = torch.utils.data.distributed.DistributedSampler(train_basis_dataset)
    else:
        train_sampler = None
        train_basis_sampler = None
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                shuffle=(train_sampler is None), num_workers=args.workers,
                                                pin_memory=True, sampler=train_sampler, drop_last=False)
        
    train_basis_loader = torch.utils.data.DataLoader(train_basis_dataset, batch_size=1,
                                                shuffle=(train_basis_sampler is None), num_workers=args.workers,
                                                pin_memory=True, sampler=train_basis_sampler, drop_last=False)


    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch, shuffle=True,
                                             num_workers=0, pin_memory=True, drop_last=False)

    val_loader = {'VAL': val_loader, 'VALSET': val_dataset, 'TRAINSET': train_unpair_dataset['FULL']}

    return train_loader, val_loader, train_sampler, train_basis_loader, train_basis_sampler



def get_loader(args, dataset, with_pair=False):
    train_dataset = dataset['train']
    val_dataset = dataset['val']

    if local_rank == 0:
        print(f'#val_samples: {len(val_dataset)}')

    train_dataset_ = train_dataset['TRAIN']
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_)
    else:
        train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(train_dataset_, batch_size=args.batch_size,
                                                shuffle=(train_sampler is None), num_workers=args.workers,
                                                pin_memory=True, sampler=train_sampler, drop_last=False)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch, shuffle=True,
                                             num_workers=0, pin_memory=True, drop_last=False)
    val_loader = {'VAL': val_loader, 'VALSET': val_dataset, 'TRAINSET': train_dataset['FULL']}

    return train_loader, val_loader, train_sampler


def map_exec_func(args):
    assert args.train_mode == 'GAN'
    trainFunc = trainGANCF if args.content_fusion else trainGAN

    return trainFunc, validateUN


def save_model(args, epoch, networks, opts):
    check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
    # if (epoch + 1) % (args.epochs//10) == 0:
    with torch.no_grad():
        save_dict = {}
        save_dict['epoch'] = epoch + 1
        for name, net in networks.items():
            save_dict[name+'_state_dict'] = net.state_dict()
            if name in ['G_EMA', 'C_EMA']:
                continue
            save_dict[name.lower()+'_optimizer'] = opts[name].state_dict()
        print("SAVE CHECKPOINT[{}] DONE".format(epoch+1))
        save_checkpoint(save_dict, check_list, args, oss_client, epoch + 1)
    check_list.close()

    if args.on_oss:
        oss_client.write_file(os.path.join(args.log_dir, "checkpoint.txt"), \
            os.path.join(args.log_dir_oss, "checkpoint.txt"))


if __name__ == '__main__':
    main()
    # if args.local_rank == 0 and args.remote:
    #     print('lazy packing...')
    #     os.system('tar -czvf {} {} > /dev/null'.format('/tmp/tmp.tgz', args.save_path))
    #     oss_client.write_file('/tmp/tmp.tgz', os.path.join(args.log_dir_oss, "tmp.tgz"))
    #     print('Done')
