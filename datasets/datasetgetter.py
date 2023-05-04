import torch
from torchvision.datasets import ImageFolder
import os
import random
import torchvision.transforms as transforms
from datasets.custom_dataset import ImageFolerRemap, CrossdomainFolder, \
    ImageFolerRemapPairCF, ImageFolerRemapUnpairCF, ImageFolerRemapPairbasis, \
    ImageFolerRemapPair, TwoDataset

class Compose(object):
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img):
        for t in self.tf:
            img = t(img)
        return img


def get_dataset(args, data_dir=None, class_to_use=None, with_path=False):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   normalize])
    transform_val = Compose([transforms.Resize((args.img_size, args.img_size)),
                                       transforms.ToTensor(),
                                       normalize])

    class_to_use = class_to_use or args.att_to_use
    remap_table = {k: i for i, k in enumerate(class_to_use)}
    if args.local_rank == 0:
        print(f'USE CLASSES: {class_to_use}\nLABEL MAP: {remap_table}')

    img_dir = data_dir or args.data_dir

    dataset = ImageFolerRemap(img_dir, transform=transform, remap_table=remap_table, with_path=with_path)
    valdataset = ImageFolerRemap(img_dir, transform=transform_val, remap_table=remap_table, with_path=with_path)
    # parse classes to use
    tot_targets = torch.tensor(dataset.targets)

    if True: # my implement
        train_dataset = {'TRAIN': dataset, 'FULL': dataset}
        subset_indices = random.sample(range(len(valdataset)), args.val_num)
        val_dataset = torch.utils.data.Subset(valdataset, subset_indices)
    else: # DG-Font style implement
        min_data, max_data = 99999999, 0
        train_idx, val_idx = None, None
        for k in class_to_use:
            tmp_idx = (tot_targets == k).nonzero(as_tuple=False)
            train_tmp_idx = tmp_idx[:-args.val_num]
            val_tmp_idx = tmp_idx[-args.val_num:]

            if k == class_to_use[0]:
                train_idx = train_tmp_idx.clone()
                val_idx = val_tmp_idx.clone()
            else:
                train_idx = torch.cat((train_idx, train_tmp_idx))
                val_idx = torch.cat((val_idx, val_tmp_idx))
            
            min_data = min(min_data, len(train_tmp_idx))
            max_data = max(max_data, len(train_tmp_idx))

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(valdataset, val_idx)
        train_dataset = {'TRAIN': train_dataset, 'FULL': dataset}
        
        args.min_data, args.max_data = min_data, max_data
        if args.local_rank == 0:
            print(f"MIN/MAX DATA: {args.min_data}/{args.max_data}")

    return train_dataset, val_dataset

def get_full_dataset_ft(args, data_dir=None, class_to_use=None, with_path=False,
        ft_ignore_target=-1):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   normalize])
    class_to_use = class_to_use or args.att_to_use
    remap_table = {k: i for i, k in enumerate(class_to_use)}
    if args.local_rank == 0:
        print(f'USE CLASSES: {class_to_use}\nLABEL MAP: {remap_table}')

    img_dir = data_dir or args.data_dir
    dataset = ImageFolerRemap(img_dir, transform=transform, remap_table=remap_table, with_path=with_path)
    dataser_ft = ImageFolerRemapPair(img_dir, transform=transform, remap_table=remap_table, ignore_target=ft_ignore_target)
    # parse classes to use
    # tot_targets = torch.tensor(dataset.targets)

    # min_data, max_data = 99999999, 0
    # train_idx, val_idx = None, None
    # for k in class_to_use:
    #     tmp_idx = (tot_targets == k).nonzero(as_tuple=False)
    #     min_data = min(min_data, len(tmp_idx))
    #     max_data = max(max_data, len(tmp_idx))
    full_dataset = {'FULL': dataset, 'FT': dataser_ft}
    # args.min_data, args.max_data = min_data, max_data
    # if args.local_rank == 0:
    #     print(f"MIN/MAX DATA: {args.min_data}/{args.max_data}")
    return full_dataset

def get_full_dataset_cfft(args, data_dir=None, base_dir=None, base_ft_dir=None, class_to_use=None, with_path=False,
        ft_ignore_target=-1, class_to_use_base=None):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   normalize])
    class_to_use = class_to_use or args.att_to_use
    class_to_use_base = class_to_use_base or args.att_to_use_base
    remap_table = {k: i for i, k in enumerate(class_to_use)}
    remap_table_base = {k: i for i, k in enumerate(class_to_use_base)}
    if args.local_rank == 0:
        print(f'USE CLASSES: {class_to_use}\nLABEL MAP: {remap_table}\nBASE LABEL MAP: {remap_table_base}')

    img_dir = data_dir or args.data_dir
    img_base_dir = base_dir or args.base_dir
    img_base_ft_dir = base_ft_dir or args.base_ft_dir
    dataset = ImageFolerRemap(img_dir, transform=transform, remap_table=remap_table, with_path=with_path)
    dataset_base = ImageFolerRemapPair(img_base_dir, transform=transform, remap_table=remap_table_base, with_path=with_path)
    dataser_full_ft = ImageFolerRemapPair(img_dir, transform=transform, remap_table=remap_table, ignore_target=ft_ignore_target)
    dataser_base_ft = ImageFolerRemapPair(img_base_ft_dir, transform=transform, remap_table=remap_table_base, ignore_target=ft_ignore_target)
    dataser_ft = TwoDataset(dataser_full_ft, dataser_base_ft)
    # parse classes to use
    tot_targets = torch.tensor(dataset.targets)

    # min_data, max_data = 99999999, 0
    # train_idx, val_idx = None, None
    # for k in class_to_use:
    #     tmp_idx = (tot_targets == k).nonzero(as_tuple=False)
    #     min_data = min(min_data, len(tmp_idx))
    #     max_data = max(max_data, len(tmp_idx))
    full_dataset = {'FULL': dataset, 'BASE': dataset_base, 'FT': dataser_ft}
    # args.min_data, args.max_data = min_data, max_data
    # if args.local_rank == 0:
    #     print(f"MIN/MAX DATA: {args.min_data}/{args.max_data}")
    return full_dataset

def get_full_dataset(args):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   normalize])
    dataset = ImageFolerRemap(args.data_dir, transform=transform, remap_table=remap_table)
    return dataset


def get_cf_dataset(args):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   normalize])

    class_to_use = args.att_to_use

    if args.local_rank == 0:
        print('USE CLASSES', class_to_use)

    # remap labels
    remap_table = {}
    i = 0
    for k in class_to_use:
        remap_table[k] = i
        i += 1

    if args.local_rank == 0:
        print("LABEL MAP:", remap_table)

    img_dir = args.data_dir

    # cf_dataset = ImageFolerRemapPairCF(img_dir, base_idxs=args.base_idxs, base_ws=args.base_ws,transform=transform, remap_table=remap_table, \
    #     sample_skip_base=True, sample_N=args.sample_N)
    cf_dataset = ImageFolerRemapUnpairCF(img_dir, base_ws=args.base_ws, transform=transform, remap_table=remap_table, top_n=args.base_top_n)
    cf_basis_dataset = ImageFolerRemapPairbasis(img_dir, base_idxs=args.base_idxs, base_ws=args.base_ws,transform=transform, remap_table=remap_table) # keep bs 1
    
    return cf_dataset, cf_basis_dataset
