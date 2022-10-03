import argparse
import os, sys
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import loss
from networks import models
import torch.nn.functional as F
import random, pdb, math, copy
from loss import CrossEntropyLabelSmooth
from collections import Counter
import matplotlib.pyplot as plt
from utils import op_copy, lr_scheduler, cal_acc, PatchMixup, CutMixup, high_scheduler, AvgLoss, RegMixup, obtain_label, score_refinement
from data_provider import data_load


def train_source(args):
    avgloss = AvgLoss()
    dset_loaders, dsets = data_load(args)
    from networks import models
    vit = models.ViT(args).cuda()
    vit.load_state_dict(torch.load(osp.join('./info/StageOne/', str(args.seed), args.dset, args.names[args.s][0].upper(),
                                            "net_{}.pt".format(mix[args.mix]))))
    if args.aux:
        vit.register_aux_classifier()
    param_group = []
    learning_rate = args.lr
    for k, v in vit.named_parameters():
        if k[:8] == 'head_cls':
            param_group += [{'params': v, 'lr': learning_rate * 0.5}]
        else:
            param_group += [{'params': v, 'lr': learning_rate * 0.1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    acc_init = 0
    max_iter = args.max_epoch * max(len(dset_loaders["source_fu"]), len(dset_loaders["target_tr"]))
    interval_iter = max_iter // args.max_epoch
    iter_num = 0
    epoch = 0
    vit.eval()
    data = None
    mem_label = torch.zeros(len(dset_loaders["target_tr"].dataset)).cuda().long()
    while iter_num < max_iter:
        try:
            inputs_source, labels_source, src_idx = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_fu"])
            inputs_source, labels_source, src_idx = iter_source.next()

        try:
            inputs_target, _, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target_tr"])
            inputs_target, _, tar_idx = iter_target.next()

        if inputs_target.size(0) == 1:
            continue
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        high = high_scheduler(epoch, args.max_epoch)
        if iter_num % (1 * interval_iter) == 0:
            vit.eval()
            cls = True
            if args.aux:
                cls = False

            if data is not None:
                mem_label, _, _ = obtain_label(dset_loaders['target_te'], vit, args, cls=cls, data=data)
            else:
                mem_label, _, _ = obtain_label(dset_loaders['target_te'], vit, args, cls=cls)
            mem_label = torch.from_numpy(mem_label).cuda()
            vit.train()
        lam = np.random.uniform(0, high)
        inputs_mix, src_labels1, tgt_labels1, lam1 = PatchMixup(src_inputs=inputs_source,
                                                                tgt_inputs=inputs_target,
                                                                src_labels=labels_source,
                                                                tgt_labels=mem_label[tar_idx], lam=lam)

        (outputs_aux, outputs_cls), _, _ = vit(inputs_mix.cuda())
        classifier_loss = lam1 * nn.CrossEntropyLoss()(outputs_cls, src_labels1.cuda()) + (
                    1 - lam1) * nn.CrossEntropyLoss()(outputs_cls, tgt_labels1.cuda())
        if args.aux:
            classifier_loss = classifier_loss + lam1 * nn.CrossEntropyLoss()(outputs_aux, src_labels1.cuda()) + (
                    1 - lam1) * nn.CrossEntropyLoss()(outputs_aux, tgt_labels1.cuda())

        avgloss.add_loss(classifier_loss.item())
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        iter_num += 1
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            epoch += 1
            vit.eval()
            avgloss_ = avgloss.get_avg_loss()
            if args.dset == 'VISDA-C':
                log_str = 'Task: {} -> {} | Iter:[{}/{}] | Loss: {:.2f} | High: {:.2f}'.format(args.names[args.s].upper(),
                                                                                             args.names[args.t].upper(),
                                                                                             epoch, args.max_epoch,
                                                                                             avgloss_, high)
            else:
                log_str = 'Task: {} -> {} | Iter:[{}/{}] | Loss: {:.2f} | High: {:.2f}'.format(args.names[args.s].upper(),
                                                                                             args.names[args.t].upper(),
                                                                                             epoch, args.max_epoch,
                                                                                             avgloss_, high)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            target = True
            data = obtain_label(dset_loaders['target_te'], vit, args, d=True)

            acc_s_te, _ = score_refinement(data[:, :args.feature_num + 1],
                                           data[:, args.feature_num + 1 + args.class_num:-1], data[:, -1], args,
                                           K=args.K)
            if args.aux:
                score_refinement(data[:, :args.feature_num + 1],
                                 data[:, args.feature_num + 1:args.feature_num + 1 + args.class_num], data[:, -1], args,
                                 K=args.K)
            vit.train()


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NCPM')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='4', help="device id to run")
    parser.add_argument('--dset', type=str, default='office', help="dataset")
    parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=72, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--s', type=int, default=0, help="source domain")
    parser.add_argument('--mix', type=int, default=0, help="mixing strategies")
    parser.add_argument('--aux', type=int, default=1, help="auxiliary classifier")
    parser.add_argument('--t', type=int, default=1, help="target domain")
    parser.add_argument('--K', type=int, default=10, help="M-nearest neighbors")
    parser.add_argument('--nn', type=int, default=2, help="maximum number of WKC")
    parser.add_argument('--feature_num', type=int, default=384, help="number of workers")
    parser.add_argument('--lr', type=float, default=3e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='deit-s', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--min', type=float, default=0.0)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='info/StageTwoAndThree/')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    args = parser.parse_args()
    mix = {-1: "ERM", 0: "Patch-wiseCutMix"}

    if args.dset == 'office-home':
        args.names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
        args.K = 10
        args.n = 2
    if args.dset == 'office':
        args.n = 2
        # args.K = 4
        args.names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        args.names = ['train', 'validation']
        args.class_num = 12
        args.K = 30
        args.n = 3
    if args.dset == 'domainnet':
        args.names = ['clipart_train', 'infograph_train', 'painting_train', 'quickdraw_train', 'real_train',
                      'sketch_train']
        args.class_num = 345
        args.min = 0
        args.K = 10
        args.n = 2
    if args.net == 'deit-b':
        args.feature_num = 768
    args.name = args.names[args.s][0].upper() + '->' + args.names[args.t][0].upper()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = 'root'

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]
    args.output_dir_src = osp.join(args.output, args.da, str(args.seed), args.dset, args.names[args.s][0].upper(), args.names[args.t][0].upper())
    args.name_src = args.names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)
    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    args.s_dset_path = folder + args.names[args.s] + '_list.txt'
    args.test_dset_path = folder + args.names[args.t] + '_list.txt'
    train_source(args)

