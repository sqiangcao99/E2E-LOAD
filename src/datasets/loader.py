#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import pdb

import itertools
import numpy as np
from functools import partial
from typing import List
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler

from src.datasets.multigrid_helper import ShortCycleBatchSampler

from . import utils as utils
from .build import build_dataset

import src.utils.distributed as du


def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """

    # pdb.set_trace()

    # inputs, labels, video_idx, time, extra_data = zip(*batch)
    x, y = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    time = [item for sublist in time for item in sublist]

    inputs, labels, video_idx, time, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(time),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, time, extra_data
    else:
        return inputs, labels, video_idx, time, extra_data

def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset

    dataset = build_dataset(dataset_name, cfg, split)
    
    # print(len(dataset))
    # exit()
    
    if cfg.NUM_GPUS>1:
        dataset = du.all_gather_unaligned(dataset)
        dataset = dataset[0]

    # all_gather_unaligned
    """
    with open('data_len'+str(du.get_rank()), 'w') as f:
        f.writelines([str(len(dataset))])
    """
    """
    if du.get_rank() == 0: 
        pdb.set_trace() # import src.utils.distributed as du
    """
    if isinstance(dataset, torch.utils.data.IterableDataset):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=detection_collate if cfg.DETECTION.ENABLE else None,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )
    else:

        # Create a sampler for multi-process training
        sampler = utils.create_sampler(dataset, shuffle, cfg)
        # Create a loader
        # if du.get_rank==0:
        #     pdb.set_trace() 

        if (
            (
                cfg.AUG.NUM_SAMPLE > 1
                or cfg.DATA.TRAIN_CROP_NUM_TEMPORAL > 1
                or cfg.DATA.TRAIN_CROP_NUM_SPATIAL > 1
            )
            and split in ["train"]
            and not cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        ):
            collate_func = partial(
                multiple_samples_collate, fold="imagenet" in dataset_name
            )
        else:
            collate_func = None
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            #pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            #collate_fn=collate_func,
            #worker_init_fn=utils.loader_worker_init_fn(dataset),
        )

    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    
    if (
        loader._dataset_kind
        == torch.utils.data.dataloader._DatasetKind.Iterable
    ):
        if hasattr(loader.dataset, "sampler"):
            sampler = loader.dataset.sampler
        else:
            raise RuntimeError(
                "Unknown sampler for IterableDataset when shuffling dataset"
            )
    else:
        sampler = (
            loader.batch_sampler.sampler
            if isinstance(loader.batch_sampler, ShortCycleBatchSampler)
            else loader.sampler
        )
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    
    # RandomSampler handles shuffling automatically 
    if isinstance(sampler, DistributedSampler): # 分布式训练的sampler; 
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch) 

    if hasattr(loader.dataset, "prefetcher"): # false; 
        sampler = loader.dataset.prefetcher.sampler
        if isinstance(sampler, DistributedSampler):
            # DistributedSampler shuffles data based on epoch
            print("prefetcher sampler")
            sampler.set_epoch(cur_epoch)