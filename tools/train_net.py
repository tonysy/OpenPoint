#
# @Author: Songyang Zhang 
# @Date: 2019-01-24 22:06:55 
# @Last Modified by:   Songyang Zhang 
# @Last Modified time: 2019-01-24 22:06:55 
#
r"""
Basic training script for PyTorch
"""

from openpoint.utils.env import setup_environment

import argparse
import os

import torch

from openpoint.config import cfg
from openpoint.data import make_data_loader 
from openpoint.solver import make_lr_scheduler
from openpoint.solver import make_optimizer


from tensorboardX import SummaryWriter

def train(cfg, local_rank, distributed):
    model = build_point_cloud_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["Epoch"] = 0

    output_dir = cfg.OUTPUT_DIR 
    if cfg.PLOT_CURVE:
        writer = SummaryWriter(output_dir)
    else:
        writer = None

    
    save_to_disk = get_rank() == 0