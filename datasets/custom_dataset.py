import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys

import torch
import random
import numpy as np
from tqdm import tqdm
import time
import cv2
import pdb

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class TwoDataset(torch.utils.data.Dataset):    
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB
        assert len(self.datasetA) == len(self.datasetB), (len(self.datasetA), len(self.datasetB))
        
    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        return xA, xB
    
    def __len__(self):
        return len(self.datasetA)

class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        classes.sort(key= lambda x:int(x[3:]))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        imgname = path.split('/')[-1].replace('.JPEG', '')
        return sample, target, imgname

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolerRemap(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, remap_table=None, with_idx=False, with_path=False):
        super(ImageFolerRemap, self).__init__(root, loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform)
        

        self.imgs = self.samples
        self.with_path = with_path
        self.class_table = remap_table
        self.with_idx = with_idx
        self.with_path = with_path

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.class_table[target]
        if self.with_idx:
            return sample, index, target # image sample_index class_index
        elif self.with_path:
            path_local = os.path.join(*path.split('/')[-2:])
            return sample, path_local, target # image sample_index class_index
        return sample, target # image class_index

class ImageFolerRemapUnpairCF(DatasetFolder):
    def __init__(self, root, base_ws, transform=None, target_transform=None,
                 loader=default_loader, remap_table=None, with_idx=False, top_n=-1):
        super(ImageFolerRemapUnpairCF, self).__init__(root, loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform)

        self.imgs = self.samples
        self.class_table = remap_table
        self.with_idx = with_idx

        # topk
        if top_n > 0:
            top_v,top_idx = base_ws.topk(top_n)
            base_ws = torch.zeros_like(base_ws).scatter_(1, top_idx, top_v)
            self.base_ws = base_ws / base_ws.sum(1, keepdim=True)
        else:
            self.base_ws = base_ws # [400, 10]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.class_table[target]

        # print(font_ids)
        base_w = self.base_ws[target].clone()

        if self.with_idx:
            return sample, index, target, base_w # image sample_index class_index
        return sample, target, base_w # image class_index

class ImageFolerRemapPair(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
            loader=default_loader, remap_table=None, sample_skip_base=False,
            ignore_target=-1, with_path=False):
        super(ImageFolerRemapPair, self).__init__(root, loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform)
        self.imgs = self.samples
        self.class_table = remap_table
        self.sample_skip_base = sample_skip_base
        self.font_idxs = [s[1] for s in self.samples]
        self.font_N = np.max(self.font_idxs) + 1
        self._get_char_idx()
        self._find_samples_from_charidx()
        self.ignore_target=ignore_target
        self.classes=sorted(self.class_table.keys())
        self.with_path = with_path

    def _get_char_idx(self):
        fonts_max = [0 for _ in range(self.font_N + 1)]
        char_idxs = []
        for _, fi in self.samples:
            char_idxs.append(fonts_max[fi])
            fonts_max[fi] += 1
        self.char_idxs = char_idxs
        self.char_idxs_max = np.max(self.char_idxs)

    def _find_samples_from_charidx(self):
        charidx2idx = [[] for _ in range(self.char_idxs_max + 1)]
        for i, charidx in enumerate(self.char_idxs):
            charidx2idx[charidx].append(i)
        self.charidx2idx = charidx2idx

    def _get_sample_from_idx(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.class_table[target]
        
        path_local = os.path.join(*path.split('/')[-2:])
        return sample, target, path_local

    def __getitem__(self, index): # char i
        basis = []
        targets = []
        paths = []
        for idx in self.classes:
            #pdb.set_trace()
            if idx == self.ignore_target: continue
            base, target, path = self._get_sample_from_idx(self.charidx2idx[index][idx])
            if target == self.ignore_target: continue
            assert target == self.font_idxs[self.charidx2idx[index][idx]]
            basis.append(base)
            targets.append(target)
            paths.append(path)
        basis = torch.stack(basis)
        targets = torch.tensor(targets)
        sort_idx = torch.argsort(targets)
        paths = np.array(paths)
        basis = basis[sort_idx]
        paths = paths[sort_idx]
        targets = targets[sort_idx]
        paths = ' '.join(paths)
        if self.with_path:
            return basis, paths, targets
        else:
            return basis, targets

    def __len__(self):
        return self.char_idxs_max + 1


class ImageFolerRemapPairbasis(DatasetFolder):
    def __init__(self, root, base_idxs, base_ws, transform=None, target_transform=None, loader=default_loader, remap_table=None, sample_skip_base=False):
        super(ImageFolerRemapPairbasis, self).__init__(root, loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform)
        self.imgs = self.samples
        self.class_table = remap_table
        self.base_idxs = base_idxs
        self.sample_skip_base = sample_skip_base
        self.font_idxs = [s[1] for s in self.samples]
        self.font_N = np.max(self.font_idxs) + 1
        self._get_char_idx()
        self._find_samples_from_charidx()

    def _get_char_idx(self):
        fonts_max = [0 for _ in range(self.font_N + 1)]
        char_idxs = []
        for _, fi in self.samples:
            char_idxs.append(fonts_max[fi])
            fonts_max[fi] += 1
        self.char_idxs = char_idxs
        self.char_idxs_max = np.max(self.char_idxs)

    def _find_samples_from_charidx(self):
        charidx2idx = [[] for _ in range(self.char_idxs_max + 1)]
        charidx2idx_basis = [[] for _ in range(self.char_idxs_max + 1)]

        self.idx2bidx = {self.base_idxs[i]:i for i in range(len(self.base_idxs))}

        for i, charidx in enumerate(self.char_idxs):
            font_idx = self.samples[i][1]
            if font_idx in self.base_idxs:
                charidx2idx_basis[charidx].append(i)
                if self.sample_skip_base:
                    continue
            charidx2idx[charidx].append(i)
        self.charidx2idx = charidx2idx
        self.charidx2idx_basis = charidx2idx_basis

    def _get_sample_from_idx(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.class_table[target]
        return sample, target

    def __getitem__(self, index): # char i
        basis = []
        for base_idx in range(len(self.base_idxs)):
            base, target = self._get_sample_from_idx(self.charidx2idx_basis[index][base_idx])
            assert target == self.font_idxs[self.charidx2idx_basis[index][base_idx]]
            basis.append(base)
        return torch.stack(basis)

    def __len__(self):
        return self.char_idxs_max

class ImageFolerRemapPairCF(DatasetFolder):
    def __init__(self, root, base_idxs, base_ws, transform=None, target_transform=None, loader=default_loader, remap_table=None, sample_skip_base=False, sample_N=10, keep_unpair=True, top_n=-1):
        super(ImageFolerRemapPairCF, self).__init__(root, loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform)
        self.imgs = self.samples
        self.class_table = remap_table
        self.base_idxs = base_idxs
        self.sample_skip_base = sample_skip_base
        self.keep_unpair = keep_unpair
        self.font_idxs = [s[1] for s in self.samples]
        self.font_N = np.max(self.font_idxs) + 1
        self._get_char_idx()
        self._find_samples_from_charidx()
        self.sample_N=sample_N
        # topk
        if top_n > 0:
            top_v,top_idx = base_ws.topk(top_n)
            base_ws = torch.zeros_like(base_ws).scatter_(1, top_idx, top_v)
            self.base_ws = base_ws / base_ws.sum(1, keepdim=True)
        else:
            self.base_ws = base_ws # [400, 10]

    def _get_char_idx(self):
        fonts_max = [0 for _ in range(self.font_N + 1)]
        char_idxs = []
        for _, fi in self.samples:
            char_idxs.append(fonts_max[fi])
            fonts_max[fi] += 1
        self.char_idxs = char_idxs
        self.char_idxs_max = np.max(self.char_idxs)

    def _find_samples_from_charidx(self):
        charidx2idx = [[] for _ in range(self.char_idxs_max + 1)]
        charidx2idx_basis = [[] for _ in range(self.char_idxs_max + 1)]

        self.idx2bidx = {self.base_idxs[i]:i for i in range(len(self.base_idxs))}

        for i, charidx in enumerate(self.char_idxs):
            font_idx = self.samples[i][1]
            if font_idx in self.base_idxs:
                charidx2idx_basis[charidx].append(i)
                if self.sample_skip_base:
                    continue
            charidx2idx[charidx].append(i)
        self.charidx2idx = charidx2idx
        self.charidx2idx_basis = charidx2idx_basis

    def _get_sample_from_idx(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.class_table[target]
        return sample, target

    def __getitem__(self, index): # char i
        unpair_n = -1 if index % 2 == 0 else self.sample_N // 2
        pair_n = self.sample_N if unpair_n == -1 else self.sample_N - unpair_n
        samples = []
        basis = []
        font_ids = []


        if unpair_n > 0:
            random_ids_unpair = random.sample(range(len(self.samples)), unpair_n)
            for random_id in random_ids_unpair:
                sample, target = self._get_sample_from_idx(random_id)
                font_ids.append(target)
                samples.append(sample)

        random_ids_pair = random.sample(range(len(self.charidx2idx[index])), pair_n)
        for random_id in random_ids_pair:
            sample, target = self._get_sample_from_idx(self.charidx2idx[index][random_id])
            assert target == self.font_idxs[self.charidx2idx[index][random_id]]
            font_ids.append(target)
            samples.append(sample)
            
        for base_idx in range(len(self.base_idxs)):
            base, target = self._get_sample_from_idx(self.charidx2idx_basis[index][base_idx])
            assert target == self.font_idxs[self.charidx2idx_basis[index][base_idx]]
            basis.append(base)
        
        # print(font_ids)
        font_ids = torch.tensor(font_ids)
        base_ws = self.base_ws[font_ids].clone()

        # print(base_ws.shape)
        # print(torch.stack(samples).device, base_ws.device)
        return torch.stack(samples), font_ids, index, torch.stack(basis), base_ws, unpair_n

    def __len__(self): # !!!
        return self.char_idxs_max

class CrossdomainFolder(data.Dataset):
    def __init__(self, root, data_to_use=['photo', 'monet'], transform=None, loader=default_loader, extensions='jpg'):
        self.data_to_use = data_to_use
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name in self.data_to_use]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and d in self.data_to_use]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

if __name__ == '__main__':
    print('hi')
    import torch
    from torchvision.datasets import ImageFolder
    import os
    import torchvision.transforms as transforms
    from datasets.custom_dataset import ImageFolerRemap, CrossdomainFolder, ImageFolerRemapPair
    class Compose(object):
        def __init__(self, tf):
            self.tf = tf

        def __call__(self, img):
            for t in self.tf:
                img = t(img)
            return img

    mean = std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = Compose([transforms.ToTensor(), normalize])

    # remap labels
    remap_table = {}
    for k in range(400):
        remap_table[k] = k

    dataset = ImageFolerRemapPairCF('data', sample_N=20, base_idxs = list(range(10)) ,transform=transform, remap_table=remap_table)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

    for samples, index, basis in tqdm(dataloader):
        print(samples.shape, index, basis.shape)
        pass
        # print(samples.shape)
        # exit()
