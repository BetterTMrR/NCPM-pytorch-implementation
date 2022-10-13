import argparse
import os, sys
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import loss
from networks import models
import random, pdb, math, copy
from utils import op_copy, lr_scheduler, cal_acc, AvgLoss, RegMixup, obtain_label
from data_provider import data_load


def train_source(args):
    avg_loss = AvgLoss()
    dset_loaders, dsets = data_load(args)
    vit = models.model_dict(args).cuda()
    max_epoch = args.max_epoch
    param_group = []
    learning_rate = args.lr
    for k, v in vit.named_parameters():
        if k[:4] == 'head' or k[:6] == 'domain':
            param_group += [{'params': v, 'lr': learning_rate}]
        else:
            param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    max_iter = max_epoch * len(dset_loaders["source_fu"])
    interval_iter = max_iter // max_epoch
    iter_num = 0
    epoch = 0
    acc_init = 0.0
    vit.train()
    while iter_num < max_iter:
        try:
            inputs_source, labels_source, src_idx = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source, src_idx = iter_source.next()

        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        if args.mix == -1:
            (outputs_src, _), _, _ = vit(inputs_source.cuda())
            classifier_loss = nn.CrossEntropyLoss()(outputs_src, labels_source.cuda())
        else:
            random_idx = torch.randperm(inputs_source.shape[0])
            lam = np.random.uniform(0, 1)
            inputs_mix, src_labels1, tgt_labels1, lam1 = RegMixup(src_inputs=inputs_source, tgt_inputs=inputs_source[random_idx], src_labels=labels_source, tgt_labels=labels_source[random_idx], path_size=args.ps, lam=lam)
            (outputs_src, _), _, _ = vit(inputs_mix.cuda())
            classifier_loss = lam1 * nn.CrossEntropyLoss()(outputs_src, src_labels1.cuda()) + (1 - lam1) * nn.CrossEntropyLoss()(outputs_src, tgt_labels1.cuda())

        loss_ = classifier_loss
        avg_loss.add_loss(loss_.item())
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        iter_num += 1

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            vit.eval()
            loss__ = avg_loss.get_avg_loss()
            epoch += 1
            acc_s_te, _ = cal_acc(args, dset_loaders['source_te'], vit)
            log_str = 'Task: {} | Loss: {:.4f} | Iter:[{}/{}] | Test = {:.2f}%'.format(mix[args.mix], loss__, epoch, max_epoch, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_net = vit.state_dict()
                torch.save(best_net, osp.join(args.output_dir_src, "net_{}.pt".format(mix[args.mix])))
            vit.train()


def test_target(args):
    dset_loaders, dsets = data_load(args)
    from networks import models
    vit = models.ViT(args, pretrain=False).cuda()
    vit.load_state_dict(torch.load(osp.join('./info/Mixup/', str(args.seed), args.dset, args.name[0],
                                            "net_{}.pt".format(mix[args.mix]))))
    vit.register_aux_classifier()
    vit.eval()
    vit.parameters()
    args.out_file.write(args.names[args.s][0].upper() + "->" + args.names[args.t][0].upper() + ': ')
    obtain_label(dset_loaders['target_te'], vit, args)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NCPM')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dset', type=str, default='office', help="data set")
    parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=72, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--s', type=int, default=0, help="source domain")
    parser.add_argument('--t', type=int, default=1, help="target domain")
    parser.add_argument('--ps', type=int, default=16, help="patch size")

    parser.add_argument('--K', type=int, default=10, help="M-nearest")
    parser.add_argument('--mix', type=int, default=0, help="mixing strategies")
    # parser.add_argument('--n', type=int, default=2, help="number of workers")
    parser.add_argument('--nn', type=int, default=2, help="ma")
    parser.add_argument('--feature_num', type=int, default=384, help="number of workers")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='deit-s', help="feature extractor")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='info/StageOne/')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    args = parser.parse_args()
    # mix = {-1: "ERM", 0: "patch-wise CutMix", 1: "CutMix", 2: "mixup"}
    mix = {-1: "ERM", 0: "Patch-wiseCutMix"}
    if args.dset == 'office-home':
        args.names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.root = '/home/jiujunhe/data/office_home/imgs/'
        args.class_num = 65
        args.K = 10
    if args.dset == 'office':
        # args.K = 4
        args.root = '/home/jiujunhe/data/office/domain_adaptation_images/'
        args.names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        args.names = ['train', 'validation']
        args.class_num = 12
        args.K = 30
        args.n = 3

    if args.dset == 'domainnet':
        args.names = ['clipart_train', 'infograph_train', 'painting_train', 'quickdraw_train', 'real_train', 'sketch_train']
        args.names_test = ['clipart_test', 'infograph_test', 'painting_test', 'quickdraw_test', 'real_test', 'sketch_test']
        args.class_num = 345
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
    folder = './data/{}/'.format(args.dset)

    args.output_dir_src = osp.join(args.output, str(args.seed), args.dset, args.names[args.s][0].upper())
    args.name_src = args.names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)
    file_name = mix[args.mix]+'_{}'.format(args.ps) if args.mix == 0 else mix[args.mix]
    args.out_file = open(osp.join(args.output_dir_src, 'log_{}.txt'.format(file_name)), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    args.s_dset_path = folder + args.names[args.s] + '_list.txt'
    args.test_dset_path = folder + args.names[args.t] + '_list.txt'
    train_source(args)
