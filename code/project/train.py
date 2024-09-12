import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.model import DeepLab
from models.loss import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.initialization import download_weights, seed_everything, show_config, worker_init_fn
from utils.fit import fit_one_epoch

# Load pretrained model
def load_pretrained_weights(model, model_path, device):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    temp_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    return model

if __name__ == "__main__":
    seed = 1
    num_classes = 19
    backbone = "mobilenet"
    pretrained = True
    model_path = "model_data/deeplab_mobilenetv2.pth"
    downsample_factor = 16
    input_shape = [512, 512]
    VOCdevkit_path = 'split'
    Image_path = 'D:/jue/Data'
    dice_loss = True
    focal_loss = True
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 4

    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0
    download_weights(backbone)

    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    if local_rank == 0:
        print(f'Load weights {model_path}.')
    model = load_pretrained_weights(model, model_path, device)

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join('logs', "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    model_train = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    with open(os.path.join(VOCdevkit_path, "train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "val.txt"), "r") as f:
        val_lines = f.readlines()

    num_train, num_val = len(train_lines), len(val_lines)
    if local_rank == 0:
        show_config(num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, Init_Epoch=0,
                    Freeze_Epoch=50, UnFreeze_Epoch=100, Freeze_batch_size=8, Unfreeze_batch_size=4, Freeze_Train=True,
                    Init_lr=7e-3, Min_lr=7e-5, optimizer_type="sgd", momentum=0.9, lr_decay_type="cos", save_period=5,
                    save_dir='logs', num_workers=4, num_train=num_train, num_val=num_val)

    UnFreeze_flag = False
    for param in model.backbone.parameters():
        param.requires_grad = False
    batch_size = 8
    nbs = 16
    Init_lr_fit = min(max(batch_size / nbs * 7e-3, 5e-4), 1e-1)
    Min_lr_fit = min(max(batch_size / nbs * 7e-5, 5e-4 * 1e-2), 1e-1 * 1e-2)
    optimizer = optim.SGD(model.parameters(), Init_lr_fit, momentum=0.9, nesterov=True, weight_decay=1e-4)
    lr_scheduler_func = get_lr_scheduler("cos", Init_lr_fit, Min_lr_fit, 100)
    epoch_step, epoch_step_val = num_train // batch_size, num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("ERROR!!! Dataset too small")

    train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, Image_path)
    val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, Image_path)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=deeplab_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=local_rank, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=local_rank, seed=seed))

    if local_rank == 0:
        eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, device.type == 'cuda', eval_flag=True, period=5)
    else:
        eval_callback = None

    for epoch in range(0, 100):
        if epoch >= 50 and not UnFreeze_flag:
            batch_size = 4
            Init_lr_fit = min(max(batch_size / nbs * 7e-3, 5e-4), 1e-1)
            Min_lr_fit = min(max(batch_size / nbs * 7e-5, 5e-4 * 1e-2), 1e-1 * 1e-2)
            lr_scheduler_func = get_lr_scheduler("cos", Init_lr_fit, Min_lr_fit, 100)
            for param in model.backbone.parameters():
                param.requires_grad = True
            epoch_step, epoch_step_val = num_train // batch_size, num_val // batch_size
            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("ERROR!!! Dataset too small")
            gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=local_rank, seed=seed))
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                 drop_last=True, collate_fn=deeplab_dataset_collate, worker_init_fn=partial(worker_init_fn, rank=local_rank, seed=seed))
            UnFreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 100, True, dice_loss, focal_loss, cls_weights, num_classes, False, None, 5, 'logs', local_rank)

    if local_rank == 0:
        loss_history.writer.close()

