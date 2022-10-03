from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from timm.data import create_transform


def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


def image_train(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    # transform = create_transform(
    #     input_size=crop_size,
    #     is_training=True,
    #     color_jitter=0.4,
    #     auto_augment=None,
    #     interpolation='bicubic',
    #     re_prob=0.25,
    #     re_mode='pixel',
    #     re_count=1,
    # )
    # return transform


def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
        txt_test = new_tar.copy()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_fu"] = ImageList_idx(txt_src, transform=image_train())
    dsets["source_fu_te"] = ImageList(txt_src, transform=image_test())
    dsets["source_tr"] = ImageList_idx(tr_txt, transform=image_train())
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dsets["target_tr"] = ImageList_idx(txt_test, transform=image_train())
    dsets["target_te"] = ImageList(txt_test, transform=image_test())

    dset_loaders["source_fu"] = DataLoader(dsets["source_fu"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders["source_fu_te"] = DataLoader(dsets["source_fu_te"], batch_size=train_bs * 5, shuffle=False, num_workers=args.worker, drop_last=False)
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs * 5, shuffle=False, num_workers=args.worker, drop_last=False)
    dset_loaders["target_tr"] = DataLoader(dsets["target_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs * 5, shuffle=False, num_workers=args.worker, drop_last=False)

    print('Length of source_fu: %s' % len(dsets['source_fu']))
    # print('Length of source_tr: %s' % len(dsets['source_tr']))
    # print('Length of source_te: %s' % len(dsets['source_te']))
    # print('Length of target_tr: %s' % len(dsets['target_tr']))
    print('Length of target_te: %s' % len(dsets['target_te']))

    return dset_loaders, dsets
