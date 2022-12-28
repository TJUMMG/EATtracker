# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# ------------------------------------------------------------------------------
import _init_paths
import os
import shutil
import time
import math
import pprint
import argparse
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.utils import build_lr_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import models.models as models
from utils.utils import create_logger, print_speed, load_pretrain, restore_from, save_model
from dataset.oceanplus_wyt import OceanPlusDataset
from core.configoceanplus_wyt import config, update_config##
from core.function import oceanplus_train

eps = 1e-5

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Train OceanPlus')##
    # general
    parser.add_argument('--cfg', type=str, default='/media/HardDisk/wyt/EATracker/experiments/train/OceanPlus_wyt.yaml', help='yaml configure file name')

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    parser.add_argument('--gpus', type=str,default='0', help='gpus')
    parser.add_argument('--workers', type=int, default=8, help=' num of dataloader workers')

    args = parser.parse_args()

    return args


def reset_config(config, args):
    """
    set gpus and workers
    """
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def check_trainable(model, logger):##日志记录可训练参数
    """
    print trainable params info
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info('trainable params:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    assert len(trainable_params) > 0, 'no trainable parameters'

    return trainable_params


def get_optimizer(cfg, trainable_params):
    """
    get optimizer
    """

    optimizer = torch.optim.SGD(trainable_params, cfg.OCEANPLUS.TRAIN.LR,
                    momentum=cfg.OCEANPLUS.TRAIN.MOMENTUM,
                    weight_decay=cfg.OCEANPLUS.TRAIN.WEIGHT_DECAY)

    return optimizer

def build_opt_lr(cfg, model, current_epoch=0):
    # fix all backbone first
    for param in model.features.features.parameters():
        param.requires_grad = False
    for m in model.features.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    ## fix neck
    for param in model.neck.parameters():
        param.requires_grad = False
    for m in model.neck.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    ## fix connect_model
    for param in model.connect_model.parameters():
        param.requires_grad = False
    for m in model.connect_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    ## fix mask
    for param in model.mask_model.parameters():
        param.requires_grad = False
    for m in model.mask_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    ##unfix mask maskedge
    # for param in model.mask_model.att.parameters():
    #     param.requires_grad = True

    for param in model.mask_model.egnet.parameters():
        param.requires_grad = True


    # for layer in cfg.OCEANPLUS.TRAIN.TRAINABLE_LAYER:
    #     for param in getattr(model.features.features, layer).parameters():
    #                     param.requires_grad = True
    #     for m in getattr(model.features.features, layer).modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.train()

    if current_epoch >= cfg.OCEANPLUS.TRAIN.UNFIX_EPOCH:
        if len(cfg.OCEANPLUS.TRAIN.TRAINABLE_LAYER) > 0:  # specific trainable layers
            for layer in cfg.OCEANPLUS.TRAIN.TRAINABLE_LAYER:
                for param in getattr(model.features.features, layer).parameters():
                    param.requires_grad = True
                for m in getattr(model.features.features, layer).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
        else:    # train all backbone layers
            for param in model.features.features.parameters():
                param.requires_grad = True
            for m in model.features.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:##执行
        for param in model.features.features.parameters():
            param.requires_grad = False
        for m in model.features.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = []
    # trainable_params += [{'params': filter(lambda x: x.requires_grad,
    #                                        model.features.features.parameters()),
    #                       'lr': cfg.OCEANPLUS.TRAIN.LAYERS_LR * cfg.OCEANPLUS.TRAIN.BASE_LR}]##0.1*0.005
    # try:
    #     trainable_params += [{'params': model.neck.parameters(),
    #                               'lr': cfg.OCEANPLUS.TRAIN.BASE_LR}]
    # except:
    #     pass
    #
    # trainable_params += [{'params': model.connect_model.parameters(),
    #                       'lr': cfg.OCEANPLUS.TRAIN.BASE_LR}]
    #
    # try:
    #     trainable_params += [{'params': model.align_head.parameters(),
    #                         'lr': cfg.OCEANPLUS.TRAIN.BASE_LR}]
    # except:
    #     pass

    # ##加载MMS或MSS里面的可学习参数
    # try:
    #     trainable_params += [{'params': model.mask_model.parameters(),
    #                         'lr': cfg.OCEANPLUS.TRAIN.BASE_LR}]
    # except:
    #     pass

    ##加载mask部分参数
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.mask_model.parameters()),
                          'lr': cfg.OCEANPLUS.TRAIN.BASE_LR}]  ##0.1*0.005

    # print trainable parameter (first check)
    print('==========first check trainable==========')
    for param in trainable_params:
        print(param)


    optimizer = torch.optim.Adam(trainable_params,
                                weight_decay=cfg.OCEANPLUS.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, cfg, epochs=cfg.OCEANPLUS.TRAIN.END_EPOCH, modelFLAG='OCEANPLUS')
    lr_scheduler.step(cfg.OCEANPLUS.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def lr_decay(cfg, optimizer):
    if cfg.OCEANPLUS.TRAIN.LR_POLICY == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=0.8685)
    elif cfg.OCEANPLUS.TRAIN.LR_POLICY == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif cfg.OCEANPLUS.TRAIN.LR_POLICY == 'Reduce':
        scheduler = ReduceLROnPlateau(optimizer, patience=5)
    elif cfg.OCEANPLUS.TRAIN.LR_POLICY == 'log':
        scheduler = np.logspace(math.log10(cfg.OCEANPLUS.TRAIN.LR), math.log10(cfg.OCEANPLUS.TRAIN.LR_END), cfg.OCEAN.TRAIN.END_EPOCH)
    else:
        raise ValueError('unsupported learing rate scheduler')

    return scheduler


def pretrain_zoo():
    GDriveIDs = dict()
    GDriveIDs['Ocean'] = "1UGriYoerXFW48_tf9R1NzwQ06M-5Yz-K"
    return GDriveIDs

def main():
    # [*] args, loggers and tensorboard

    args = parse_args()
    reset_config(config, args)

    logger, _, tb_log_dir = create_logger(config, 'OCEANPLUS', 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
    }

    # [*] gpus parallel and model prepare
    # prepare pretrained model -- download from google drive
    # auto-download train model from GoogleDrive
    if not os.path.exists('./pretrain'):
        os.makedirs('./pretrain')
    # try:
    #     DRIVEID = pretrain_zoo()
    #
    #     if not os.path.exists('./pretrain/{}'.format(config.OCEAN.TRAIN.PRETRAIN)):
    #         os.system(
    #             'wget --no-check-certificate \'https://drive.google.com/uc?export=download&id={0}\' -O ./pretrain/{1}'
    #             .format(DRIVEID[config.OCEAN.TRAIN.MODEL], config.OCEAN.TRAIN.PRETRAIN))
    # except:
    #     print('auto-download pretrained model fail, please download it and put it in pretrain directory')

    # if config.OCEANPLUS.TRAIN.ALIGN:##是否训练对齐网络
    #     print('====> train object-aware version <====')
    #     model = models.__dict__[config.OCEANPLUS.TRAIN.MODEL](align=True).cuda()  # build model
    # else:
    #     print('====> Default: train without object-aware, also prepare for OceanPlus <====')
    #     model = models.__dict__[config.OCEANPLUS.TRAIN.MODEL](align=False).cuda()  # build model
    ##是否训练多阶段网络
    if config.OCEANPLUS.TRAIN.MMS:##执行
        print('====> train MMS version <====')
        model = models.__dict__[config.OCEANPLUS.TRAIN.MODEL](mms=True).cuda()  # build model
    else:
        print('====> Default: train without object-aware, also prepare for OceanPlus <====')
        model = models.__dict__[config.OCEANPLUS.TRAIN.MODEL](mms=False).cuda()  # build model

    print(model)
    ##模型加载
    model = load_pretrain(model, '/media/HardDisk/wyt/EATracker/pretrain/{0}'.format(config.OCEANPLUS.TRAIN.PRETRAIN))    # load pretrain

    # get optimizer
    if not config.OCEANPLUS.TRAIN.START_EPOCH == config.OCEANPLUS.TRAIN.UNFIX_EPOCH:##执行,START_EPOCH=0, UNFIX_EPOCH=10
        optimizer, lr_scheduler = build_opt_lr(config, model, config.OCEANPLUS.TRAIN.START_EPOCH)## yaml文件里面的START_EPOCH为0好像两个执行效果是一样的
    else:
        optimizer, lr_scheduler = build_opt_lr(config, model, 0)  # resume wrong (last line)

    # check trainable again
    print('==========double check trainable==========')
    trainable_params = check_trainable(model, logger)           # print trainable params info
    ##类似于断点续训，没用到
    if config.OCEANPLUS.TRAIN.RESUME and config.OCEANPLUS.TRAIN.START_EPOCH != 0:   # resume ##False
        model, optimizer, args.start_epoch, arch = restore_from(model, optimizer, config.OCEANPLUS.TRAIN.RESUME)

    # parallel
    gpus = [int(i) for i in config.GPUS.split(',')]
    gpu_num = len(gpus)
    logger.info('GPU NUM: {:2d}'.format(len(gpus)))

    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model, device_ids=gpus).to(device)

    logger.info(lr_scheduler)
    logger.info('model prepare done')

    # [*] train

    for epoch in range(config.OCEANPLUS.TRAIN.START_EPOCH, config.OCEANPLUS.TRAIN.END_EPOCH):
        # build dataloader, benefit to tracking
        train_set = OceanPlusDataset(config)
        train_loader = DataLoader(train_set, batch_size=config.OCEANPLUS.TRAIN.BATCH * gpu_num, num_workers=config.WORKERS, pin_memory=True, sampler=None, drop_last=True)

        # check if it's time to train backbone
        if epoch == config.OCEANPLUS.TRAIN.UNFIX_EPOCH:
            logger.info('training backbone')
            optimizer, lr_scheduler = build_opt_lr(config, model.module, epoch)
            print('==========double check trainable==========')
            check_trainable(model, logger)  # print trainable params info

        lr_scheduler.step(epoch)
        curLR = lr_scheduler.get_cur_lr()


        model, writer_dict = oceanplus_train(train_loader, model, optimizer, epoch + 1, curLR, config, writer_dict, logger, device=device)

        # save model
        save_model(model, epoch, optimizer, config.OCEANPLUS.TRAIN.MODEL, config, isbest=False)

    writer_dict['writer'].close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
    main()




