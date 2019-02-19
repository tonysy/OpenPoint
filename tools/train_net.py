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
from openpoint.engine.inference import inference
from openpoint.engine.trainer import do_train
from openpoint.modelig.detector import build_point_cloud_model
from openpoint.utils.checkpoint import OpenPointCheckpointer
from openpoint.utils.collect_env import collect_env_info
from openpoint.utils.comm import synchronize, get_rank
from openpoint.utils.imports import import_file
from openpoint.utils.logger import setup_logger
from openpoint.utils.miscellaneous import mkdir

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
            # this shold be removed if we update BatchNorm stats
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
    checkpointer = OpenPointCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        writer
    )

    return model

def test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    