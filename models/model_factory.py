"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from models.deeplab import Res50_Deeplab, Res101_Deeplab
from models.deeplab_2branch import Res50_Deeplab_2branch, Res101_Deeplab_2branch
from torch.utils import model_zoo

from models.unet import UNet

PRETRAINED_MODEL = {
    'resnet-101-caffe': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth',
    'resnet-50-caffe':  'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet50-caffe-22edcc82.pth'
}


def model_generator(args, add_bg_mask=True):
    add_bg_mask = int(add_bg_mask)
    restore_from = args.restore_from
    # create network
    if args.model == 'DeepLab':
        model = Res101_Deeplab(num_classes=args.num_parts+add_bg_mask)
        if restore_from is None:
            restore_from = PRETRAINED_MODEL['resnet-101-caffe']
    if args.model == 'UNet':
        model = UNet(n_in_channels=3, n_out_channels=args.num_parts+add_bg_mask, n_layers=4)
        restore_from = 'None'
    elif args.model == 'DeepLab50':
        model = Res50_Deeplab(num_classes=args.num_parts+add_bg_mask)
        if restore_from is None:
            restore_from = PRETRAINED_MODEL['resnet-50-caffe']
    elif args.model == 'DeepLab101_2branch':
        model = Res101_Deeplab_2branch(num_classes=args.num_parts+add_bg_mask)
        if restore_from is None:
            restore_from = PRETRAINED_MODEL['resnet-101-caffe']
    elif args.model == 'DeepLab50_2branch':
        model = Res50_Deeplab_2branch(num_classes=args.num_parts+add_bg_mask)
        if restore_from is None:
            restore_from = PRETRAINED_MODEL['resnet-50-caffe']
    else:
        print('Model "{}" not exist!'.format(args.model))

    # load pretrained parameters
    if restore_from != 'None':
        print('load model from {}'.format(restore_from))
        if restore_from[:4] == 'http':
            saved_state_dict = model_zoo.load_url(restore_from)
        else:
            saved_state_dict = torch.load(restore_from)
        optimizer_state_dict = None
        if 'optimizer_state_dict' in saved_state_dict:
            optimizer_state_dict = saved_state_dict['optimizer_state_dict']
        if 'model_state_dict' in saved_state_dict:
            # Trainer.save_model saves dict
            saved_state_dict = saved_state_dict['model_state_dict']

        saved_state_dict = dict([(k.replace('module.', ''), v) for k, v in saved_state_dict.items()])
        # only copy the params that exist in current model (caffe-like)
        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                new_params[name].copy_(saved_state_dict[name])
        model.load_state_dict(new_params)

    return model, optimizer_state_dict


def teacher_generator(args, add_bg_mask=True):
    add_bg_mask = int(add_bg_mask)
    restore_teacher_from = args.restore_teacher_from
    # create network
    model = Res101_Deeplab_2branch(num_classes=args.num_parts+add_bg_mask)
    if restore_teacher_from is None:
        restore_teacher_from = PRETRAINED_MODEL['resnet-101-caffe']

    # load pretrained parameters
    print('load model from {}'.format(restore_teacher_from))
    saved_state_dict = torch.load(restore_teacher_from)
    if 'model_state_dict' in saved_state_dict:
        # Trainer.save_model saves dict
        saved_state_dict = saved_state_dict['model_state_dict']

    model.load_state_dict(saved_state_dict)

    return model