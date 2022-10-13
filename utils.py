import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import numpy as np
import random


class AvgLoss:
    def __init__(self):
        self.loss = 0.0
        self.n = 1e-8

    def add_loss(self, loss):
        self.loss += loss
        self.n += 1

    def get_avg_loss(self):
        avg_loss = self.loss / self.n
        self.n = 1e-8
        self.loss = 0.0
        return avg_loss


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def high_scheduler(epoch, max_epoch):
    if epoch / max_epoch <= .3:
        high = .7
    elif .3 < epoch / max_epoch < 0.4:
        high = .7
    else:
        high = 1.
    return high


def cal_acc(args, loader, net, flag=False, target=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if target:
                (_, outputs), feats, _ = net(inputs)
            else:
                (outputs, _), feats, _ = net(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, _


def obtain_label(loader, net, args, d=False, cls=False, data=None, normalize=True):
    if data is None:
        start_test = True
        # features = torch.zeros(12, )
        features = []
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                (outputs_aux, outputs_cls), feats, _ = net(inputs)
                if start_test:
                    all_fea = feats.float().cpu()
                    all_output_aux = outputs_aux.float().cpu()
                    all_output_cls = outputs_cls.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feats.cpu().float()), 0)
                    all_output_aux = torch.cat((all_output_aux, outputs_aux.float().cpu()), 0)
                    all_output_cls = torch.cat((all_output_cls, outputs_cls.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        all_output_aux = nn.Softmax(dim=1)(all_output_aux)
        all_output_cls = nn.Softmax(dim=1)(all_output_cls)
        if args.distance == 'cosine':
            all_fea_ = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            feature_normalized = (all_fea_.t() / torch.norm(all_fea_, p=2, dim=1)).t()
    else:
        assert data.shape[1] == 2 + args.feature_num + 2 * args.class_num
        feature_normalized, all_output_aux, all_output_cls, all_label = data[:, :args.feature_num+1], data[:, args.feature_num+1: args.feature_num+1+args.class_num], data[:, args.feature_num+1+args.class_num:-1], data[:, -1]
    if d:
        if not normalize:
            feature_normalized = all_fea
        data = np.concatenate([feature_normalized.numpy(), all_output_aux.numpy(), all_output_cls.numpy(), all_label.numpy()[:, None]], axis=1)
        return torch.from_numpy(data)
    if cls:
        all_output = all_output_cls.clone()
    else:
        all_output = all_output_aux.clone()
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    prob, _ = all_output.max(dim=1)
    # ambiguous_idx = torch.where(prob < min(prob.mean(), torch.Tensor([args.min])))[0]
    # all_output[ambiguous_idx.squeeze()] = 0.0
    label_ = torch.eye(all_output.size(1))
    for i in range(args.nn):
        initCenter = all_output.t().mm(feature_normalized) / (1e-8 + all_output.sum(dim=0)[:, None])
        # cosine_simi_ = feature_normalized.mm(initCenter.t())
        cosine_simi_ = torch.cosine_similarity(feature_normalized.unsqueeze(1), initCenter.unsqueeze(0), dim=-1)
        # cosine_simi_ = torch.from_numpy(- cdist(feature_normalized.numpy(), initCenter.numpy()))
        pred_label = cosine_simi_.argmax(1)
        all_output = label_[pred_label]
        acc = (pred_label.squeeze().float() == all_label.squeeze().float()).sum() / all_label.size(0)
        log_str = 'Accuracy = {:.2f}% -> Accuracy = {:.2f}%'.format(accuracy * 100, acc * 100)
        args.out_file.write(log_str + "    ")
        args.out_file.flush()
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    return pred_label.numpy().astype('int'), feature_normalized.numpy(), all_label.numpy()


def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    if W - cut_w == 0:
        cx, cy = 0, 0
    else:
        cx = np.random.randint(W - cut_w)
        cy = np.random.randint(H - cut_h)

    # bbx1 = np.clip(cx - cut_w // 2, 0, W)
    # bby1 = np.clip(cy - cut_h // 2, 0, H)
    # bbx2 = np.clip(cx + cut_w // 2, 0, W)
    # bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbx1 = cx
    bby1 = cy
    bbx2 = cx + cut_w
    bby2 = cy + cut_h

    return bbx1, bby1, bbx2, bby2


def CutMixup(src_inputs, tgt_inputs, src_labels, tgt_labels, lam=0, patch=False):
    max_batch = max(src_inputs.size(0), tgt_inputs.size(0))
    if src_inputs.size(0) < max_batch:
        repeat_times = int(np.ceil(max_batch / src_inputs.size(0)))
        src_inputs = src_inputs.repeat(repeat_times, 1, 1, 1)[:max_batch]
        src_labels = src_labels.repeat(repeat_times)[:max_batch]
    if tgt_inputs.size(0) < max_batch:
        repeat_times = int(np.ceil(max_batch / tgt_inputs.size(0)))
        tgt_inputs = tgt_inputs.repeat(repeat_times, 1, 1, 1)[:max_batch]
        tgt_labels = tgt_labels.squeeze().repeat(repeat_times)[:max_batch]
    bbx1, bby1, bbx2, bby2 = rand_bbox(src_inputs.size(), 1 - lam)
    src_inputs[:, :, bbx1:bbx2, bby1:bby2] = tgt_inputs[:, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (src_inputs.size()[-1] * src_inputs.size()[-2]))
    return src_inputs, src_labels, tgt_labels, lam


def RegMixup(src_inputs=None, tgt_inputs=None, src_labels=None, tgt_labels=None, path_size=16, lam=0):
    B_src, C, H, W = src_inputs.shape
    B_tgt, _, _, _ = tgt_inputs.shape
    max_batch = max(B_src, B_tgt)
    if B_src < max_batch:
        repeat_times = int(np.ceil(max_batch / src_inputs.size(0)))
        src_labels = src_labels.repeat(repeat_times)[:max_batch]
        src_inputs = src_inputs.repeat(repeat_times, 1, 1, 1)[:max_batch]

    if B_tgt < max_batch:
        repeat_times = int(np.ceil(max_batch / tgt_inputs.size(0)))
        tgt_inputs = tgt_inputs.repeat(repeat_times, 1, 1, 1)[:max_batch]
        if len(tgt_labels.size()) == 2:
            tgt_labels = tgt_labels.repeat(repeat_times, 1)[:max_batch]
        else:
            tgt_labels = tgt_labels.repeat(repeat_times)[:max_batch]
    size = (H // path_size, W // path_size)
    bbx1, bby1, bbx2, bby2 = rand_bbox(size, 1 - lam)
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (size[-1] * size[-2])
    src_inputs[:, :, bbx1 * path_size: bbx2 * path_size, bby1 * path_size: bby2 * path_size] = tgt_inputs[:, :,
                                                                                               bbx1 * path_size:bbx2 * path_size,
                                                                                               bby1 * path_size:bby2 * path_size]
    return src_inputs, src_labels, tgt_labels, lam


def PatchMixup(src_inputs=None, tgt_inputs=None, src_labels=None, tgt_labels=None, path_size=16, lam=0):
    B_src, C, H, W = src_inputs.shape
    B_tgt, _, _, _ = tgt_inputs.shape
    max_batch = max(B_src, B_tgt)
    if B_src < max_batch:
        repeat_times = int(np.ceil(max_batch / src_inputs.size(0)))
        src_labels = src_labels.repeat(repeat_times)[:max_batch]
        src_inputs = src_inputs.repeat(repeat_times, 1, 1, 1)[:max_batch]
    if B_tgt < max_batch:
        repeat_times = int(np.ceil(max_batch / tgt_inputs.size(0)))
        if len(tgt_labels.size()) == 2:
            tgt_labels = tgt_labels.repeat(repeat_times, 1)[:max_batch]
        else:
            tgt_labels = tgt_labels.repeat(repeat_times)[:max_batch]

        tgt_inputs = tgt_inputs.repeat(repeat_times, 1, 1, 1)[:max_batch]

    new_size = (B_src, C, H // path_size, W // path_size)
    src_inputs = src_inputs.reshape(max_batch, C, H // path_size, path_size, W // path_size, path_size).permute(0, 1, 2, 4, 3, 5).reshape(max_batch, C, -1, path_size, path_size)
    tgt_inputs = tgt_inputs.reshape(max_batch, C, H // path_size, path_size, W // path_size, path_size).permute(0, 1, 2, 4, 3, 5).reshape(max_batch, C, -1, path_size, path_size)
    sampled_num_patches = int(lam * H * W / path_size**2)
    random_idx = random.sample([i for i in range(int(H * W / path_size**2))], sampled_num_patches)
    src_inputs[:, :, random_idx, :, :] = tgt_inputs[:, :, random_idx, :, :]
    src_inputs = src_inputs.reshape(max_batch, C, H // path_size, W // path_size, path_size, path_size).permute(0, 1, 2, 4, 3, 5).reshape(max_batch, C, H, W)
    return src_inputs, src_labels, tgt_labels, 1 - lam


def score_refinement(feature_normalized, score, label, args, K=2):
    # assert data.shape[1] == 2 + args.feature_num + args.class_num
    # feature_normalized, score, label = data[:, :args.feature_num+1], data[:, args.feature_num+1:-1], data[:, -1]
    idx_tgt = torch.LongTensor([i for i in range(feature_normalized.size(0))])
    cosine_simi = feature_normalized.mm(feature_normalized.t())
    _, nearest_neighbors_idx = torch.topk(cosine_simi, k=K, largest=True)  # n x 6
    nearest_neighbors_idx = nearest_neighbors_idx[:, 0:]
    kk = (nearest_neighbors_idx[nearest_neighbors_idx] == idx_tgt.unsqueeze(-1).unsqueeze(-1)).sum(2)
    affinity = torch.where(kk > 0, 1 * torch.ones(kk.size()), 0.1 * torch.ones(kk.size()))
    n = args.n
    alpha = 0.3
    refined_score_bytgt = score.clone()
    refined_score = 0
    for i in range(n):
        refined_score_bytgt = affinity.unsqueeze(1).bmm(refined_score_bytgt[nearest_neighbors_idx]).squeeze()
        refined_score += alpha ** (n - i - 1) * refined_score_bytgt
    # refined_score = affinity.unsqueeze(1).bmm(score[nearest_neighbors_idx]).squeeze()
    # refined_score += affinity.unsqueeze(1).bmm(refined_score[nearest_neighbors_idx]).squeeze()
    pred1 = score.argmax(1)
    pred2 = refined_score.argmax(1)
    ex = ''
    if args.dset == "VISDA-C":
        matrix1 = confusion_matrix(label.squeeze().float(), torch.squeeze(pred1).float())
        matrix2 = confusion_matrix(label.squeeze().float(), torch.squeeze(pred2).float())
        acc1_ = matrix1.diagonal() / matrix1.sum(axis=1) * 100
        acc2_ = matrix2.diagonal() / matrix2.sum(axis=1) * 100
        acc1 = acc1_.mean()
        acc2 = acc2_.mean()
        aa1 = [str(np.round(i, 2)) for i in acc1_]
        aa2 = [str(np.round(i, 2)) for i in acc2_]
        acc_1 = ' '.join(aa1)
        acc_2 = ' '.join(aa2)
        ex += '\n' + acc_1 + '\n' + acc_2
    else:
        acc1 = 100 * (pred1.squeeze().float() == label.squeeze().float()).sum() / label.size(0)
        acc2 = 100 * (pred2.squeeze().float() == label.squeeze().float()).sum() / label.size(0)
    log_str = "Original Score Accuracy: {:.2f}%; Refined Score Accuracy: {:.2f}%".format(acc1, acc2) + ex
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')
    return acc1, acc2