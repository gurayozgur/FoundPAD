import argparse
import os
import random
import sys
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.join(os.getcwd()))

from backbone import get_model, get_output_dim
from data import get_dataset
from data.loaders import (ApplyWeightedRandomSampler, DataLoader, DataLoaderX,
                          DistributedWeightedSampler)
from data.transform import transform_image
from finetuning import apply_lora_model
from training import get_header, get_trainer
from utils.logging import TrainingLogger

torch.backends.cudnn.benchmark = True

os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout

def main(cfg):
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=7200000))
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # Logging
    TrainingLogger(local_rank, cfg.output_path)

    # Transform init
    transform = transform_image(
        image_size=cfg.image_size,
        normalize_type=cfg.normalize_type,
        horizontal_flip=cfg.horizontal_flip,
        rand_augment=cfg.rand_augment,
        interpolation_type=cfg.interpolation_type
    )
    transform_val = transform_image(
        image_size=cfg.image_size,
        normalize_type=cfg.normalize_type,
        interpolation_type=cfg.interpolation_type
    )

    # Dataset
    trainset, testset = get_dataset(local_rank, transform, **cfg)
    train_sampler = DistributedWeightedSampler(cfg.dataset_path) # torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True) #DistributedWeightedSampler(cfg.dataset_path)  #
    dataloader = DataLoaderX(
        local_rank=local_rank,
        dataset=trainset,
        batch_size=cfg.batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=0,
        sampler=train_sampler,
    )

    test_sampler = []
    test_dataloader = []
    for test in testset:
        test_sampler.append(torch.utils.data.distributed.DistributedSampler(test, shuffle=False))
        test_dataloader.append(DataLoaderX(
            local_rank=local_rank, dataset=test, batch_size=cfg.batch_size,
            pin_memory=True, drop_last=False, num_workers=0, sampler=test_sampler[-1]
        ))

    # Model
    model = get_model(local_rank, **cfg)
    if cfg.use_lora: # LoRA
        apply_lora_model(local_rank, model, **cfg)
    elif cfg.train_scratch:
        model.backbone.initialize_parameters()
        print("Model initialized from scratch")
    model = DistributedDataParallel(module=model.backbone, broadcast_buffers=False, device_ids=[local_rank], find_unused_parameters=False)
    if cfg.use_lora or cfg.train_scratch:
        model.train()

    # Header
    output_dim = get_output_dim(**cfg)
    header = get_header(rank=local_rank, backbone_out_dim=output_dim, **cfg).to(local_rank)
    header = DistributedDataParallel(module=header, broadcast_buffers=False, device_ids=[local_rank], find_unused_parameters=False)
    header.train()

    if cfg.training_type == "PAD_training" or cfg.training_type == "PAD_training_scratch":
        traintype = "PAD_training"
    elif cfg.training_type == "PAD_training_only_header":
        traintype = "PAD_training_only_header"
    else:
        ValueError()

    # Training
    model_trainer = get_trainer(
        rank=local_rank,
        world_size=world_size,
        model_name=cfg.model_name,
        model=model,
        transform=transform_val,
        trainset=trainset,
        dataloader=dataloader,
        train_sampler=train_sampler,
        training_type=traintype,
        config=cfg,
        header=header,
        test_dataloader=test_dataloader,
        test_sampler=test_sampler,
    )
    model_trainer.start_training()

    if local_rank == 0:
        destroy_process_group()
