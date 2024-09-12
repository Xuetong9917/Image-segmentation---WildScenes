import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# Cross entropy loss function
def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.view(n, c, -1).transpose(1, 2).contiguous().view(-1, c)
    temp_target = target.view(-1)

    return nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)


# Focal loss function
def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.view(n, c, -1).transpose(1, 2).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    loss = -((1 - pt) ** gamma) * logpt
    if alpha is not None:
        loss *= alpha
    return loss.mean()


# Dice loss function
def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs, dim=1).view(n, c, -1).transpose(1, 2).contiguous().view(n, -1, c)
    temp_target = target.view(n, -1, ct)

    tp = torch.sum(temp_target[..., :-1] * temp_inputs, dim=[0, 1])
    fp = torch.sum(temp_inputs, dim=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], dim=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    return 1 - torch.mean(score)


# Weight initialization function
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        if isinstance(m, nn.Conv2d):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    print(f'Initialize network with {init_type} type')
    net.apply(init_func)


# Learning rate scheduling function
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1, no_aug_iter_ratio=0.3, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * (iters / float(warmup_total_iters)) ** 2 + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)))
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        n = iters // step_size
        return lr * (decay_rate ** n)

    if lr_decay_type == "cos":
        warmup_total_iters = max(int(warmup_iters_ratio * total_iters), 1)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = max(int(no_aug_iter_ratio * total_iters), 1)
        lr_scheduler_func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters // step_num
        lr_scheduler_func = partial(step_lr, lr, decay_rate, step_size)

    return lr_scheduler_func


# Set the learning rate of the optimizer
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
