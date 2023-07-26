import pdb

from itertools import chain

import einops
import math
import logging
import numpy as np
import os
import random
import time
from collections import defaultdict
import cv2
import torch
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms
import torchvision.io as io

from src.utils.env import pathmgr

from . import transform as transform

from .random_erasing import RandomErasing
from .transform import create_random_augment
from . import transform as transform
from . import utils as utils

from bisect import bisect_right

logger = logging.getLogger(__name__)


def sequence_sampler(cfg, start, end, mode):
    # long history; 

    if mode == "train":  # only for training the dataset!
        if cfg.TRAIN.SEQUENCE_SAMPLER == "interpolation":
            indices = np.linspace(start, end, cfg.MODEL.LONG_MEMORY_NUM_SAMPLES)
            indices = np.sort(indices).astype(np.int32) 
            long_key_padding_mask = np.zeros(indices.shape[0]) 
            last_zero = bisect_right(indices, 0) - 1 

            if last_zero > 0: 
                long_key_padding_mask[:last_zero] = float("-inf")
            
            return indices, long_key_padding_mask

        elif cfg.TRAIN.SEQUENCE_SAMPLER == 'temporal_jitter':
            assert cfg.AUG.TEMPORAL.HISTORY.RANGE > (end - start)
            # start = max(end - cfg.AUG.TEMPORAL.HISTORY.RANGE, 0)
            # indices = np.linspace(start, end, cfg.AUG.TEMPORAL.HISTORY.RANGE)
            start = end - cfg.AUG.TEMPORAL.HISTORY.RANGE
            indices = np.arange(start, end + 1).clip(0)
            indices = np.random.choice(indices, cfg.MODEL.LONG_MEMORY_NUM_SAMPLES, replace=False)
            indices = np.sort(indices).astype(np.int32) 
            
            long_key_padding_mask = np.zeros(indices.shape[0]) 

            last_zero = bisect_right(indices, 0) - 1  
            
            if last_zero > 0: 
                long_key_padding_mask[:last_zero] = float("-inf")

            return indices, long_key_padding_mask
        
        elif cfg.TRAIN.SEQUENCE_SAMPLER == "uniform":
            indices = np.arange(start, end + 1)[
                :: cfg.MODEL.LONG_MEMORY_SAMPLE_RATE
            ]   
            if cfg.TRAIN.SEQUENCE_PADDING == "zero":
                if cfg.TEST.SEQUENCE_PADDING == "zero":
                    padding = (
                        cfg.MODEL.LONG_MEMORY_NUM_SAMPLES - indices.shape[0]
                    )
                    if padding > 0:
                        indices = np.concatenate((np.zeros(padding), indices))

                    indices = np.sort(indices).astype(np.int32)
                    long_key_padding_mask = np.zeros(indices.shape[0])
                    last_zero = bisect_right(indices, 0) - 1

                    if last_zero > 0:
                        long_key_padding_mask[:last_zero] = float("-inf")
                    return indices, long_key_padding_mask

            elif cfg.TRAIN.SEQUENCE_PADDING == "repeat": 
                raise NotImplementedError 
    else:  # for validation and testing;
        
        indices = np.arange(start, end + 1)[:: cfg.MODEL.LONG_MEMORY_SAMPLE_RATE]
        if cfg.TEST.SEQUENCE_PADDING == "zero":
            padding = cfg.MODEL.LONG_MEMORY_NUM_SAMPLES - indices.shape[0]
            if padding > 0:
                indices = np.concatenate((np.zeros(padding), indices))

            indices = np.sort(indices).astype(np.int32)
            long_key_padding_mask = np.zeros(indices.shape[0])
            last_zero = bisect_right(indices, 0) - 1

            if last_zero > 0:
                long_key_padding_mask[:last_zero] = float("-inf")
            return indices, long_key_padding_mask 


def load_image_lists(
    cfg, sessions, video_root, target_root, mode, return_list=False
):

    work_image_paths = defaultdict(list)
    long_image_paths = defaultdict(list)
    work_targets = defaultdict(list)

    # Temporal Jitter Augmentation
    if cfg.AUG.TEMPORAL.JITTER.ENABLE and mode == "train":
        max_extend_chunks = int(cfg.AUG.TEMPORAL.JITTER.MAX_EXTEND * cfg.MODEL.WORK_MEMORY_NUM_SAMPLES)
        temporal_jitter = True
    else:
        max_extend_chunks = 0
        temporal_jitter = False 
    
    # Temporal Smooth Augmentation 
    if cfg.AUG.TEMPORAL.SMOOTH.ENABLE and mode == "train":
        max_roll_chunks = int(cfg.AUG.TEMPORAL.SMOOTH.MAX_ROLL)
        temporal_smooth = True
    else:
        max_roll_chunks = 0
        temporal_smooth = False

    # chunk_shrink = max_extend_chunks + max_roll_chunks
    # chunk_shrink = 0
    chunk_shrink = max_roll_chunks
    window_stride = cfg.MODEL.WORK_MEMORY_NUM_SAMPLES if mode in ['train', 'val'] else 1

    for session in sessions:
        video_path = os.path.join(video_root, session) 
        target_path = os.path.join(target_root, session + ".npy") 
        target = np.load(target_path) 
        
        frame_length = len(os.listdir(video_path))
        frame_indices = np.arange(frame_length)

        ## Split the Video List into chunks;
        num_chunks = int(frame_length // cfg.MODEL.CHUNK_SIZE) 
        assert num_chunks == target.shape[0]

        if cfg.MODEL.CHUNK_SIZE > 1:
            chunk_indices = np.split(
                frame_indices[: num_chunks * cfg.MODEL.CHUNK_SIZE],
                num_chunks,
                axis=0,
            ) 
            chunk_indices = np.stack(chunk_indices, axis=0) 
            chunk_indices = chunk_indices[:, cfg.MODEL.CHUNK_SAMPLE_RATE // 2 :: cfg.MODEL.CHUNK_SAMPLE_RATE]
        else:
            chunk_indices = frame_indices[:,None]

        seed = (
            np.random.randint(cfg.MODEL.WORK_MEMORY_NUM_SAMPLES)
            if mode == "train" 
            else 0
        )
        
        for work_start, work_end in zip(
            # range(seed, num_chunks - chunk_shrink + 1, window_stride),
            range(seed, num_chunks - chunk_shrink, window_stride),
            range(
                seed + cfg.MODEL.WORK_MEMORY_NUM_SAMPLES,
                num_chunks - chunk_shrink,
                window_stride, # 32; 
            ),
        ):  
            if cfg.AUG.TEMPORAL.JITTER.ENABLE and mode == "train":
                if work_end >= num_chunks - max_extend_chunks - 1: # to keep the safe here; 
                    temporal_jitter = False
                else:
                    temporal_jitter = True 

            if temporal_jitter and random.random() < cfg.AUG.TEMPORAL.JITTER.RATE:
                entend_chunk = random.randint(1, max_extend_chunks)
                work_end += entend_chunk

                work_indices = np.arange(work_start, work_end).clip(0)[
                    :: cfg.MODEL.WORK_MEMORY_SAMPLE_RATE
                ] 

                if cfg.AUG.TEMPORAL.JITTER.SAMPLE_TYPE == 'random':
                    work_indices = np.random.choice(work_indices, cfg.MODEL.WORK_MEMORY_NUM_SAMPLES, replace=False) 
                    work_indices = np.sort(work_indices)
                elif cfg.AUG.TEMPORAL.JITTER.SAMPLE_TYPE == 'uniform':
                    idcs = np.linspace(0, len(work_indices)-1, cfg.MODEL.WORK_MEMORY_NUM_SAMPLES).astype(np.int32)
                    work_indices = work_indices[idcs]
            else:
                work_indices = np.arange(work_start, work_end).clip(0)[
                    :: cfg.MODEL.WORK_MEMORY_SAMPLE_RATE
                ]
            
            if temporal_smooth:
                roll_flag = -1 
                roll_chunk = 0

                if random.random() < cfg.AUG.TEMPORAL.SMOOTH.RATE:
                    roll_chunk = random.randint(1, max_roll_chunks)
                    
                    if random.random()< 0.5:
                        roll_flag = 0 # 
                    elif random.random()>= 0.5:
                        roll_flag = 1

            work_frames = []
            work_target = []

            if cfg.AUG.CURRENT.SHUFFLE.ENABLE and random.random() < cfg.AUG.CURRENT.SHUFFLE.RATE: 
                
                start_point = random.randint(0, work_indices.shape[0] - cfg.AUG.CURRENT.SHUFFLE.WIN_SIZE-1)
                local_window = work_indices[start_point:start_point+cfg.AUG.CURRENT.SHUFFLE.WIN_SIZE]
                np.random.shuffle(local_window)
                work_indices[start_point:start_point+cfg.AUG.CURRENT.SHUFFLE.WIN_SIZE] = local_window

            for indice in work_indices:
                
                findice = indice
                tindice = indice

                if temporal_smooth:
                    if roll_flag == 1:
                        findice = indice + roll_chunk
                        
                    elif roll_flag == 0:
                        tindice = indice + roll_chunk

                for frame in chunk_indices[findice]: 
                    if cfg.DATA.PATH_PREFIX == "hdd":
                        frame += 1
                    fname = cfg.DATA.FRAME_TEMPL.format(frame)
                    path = os.path.join(video_path, fname)
                    work_frames.append(path)

                work_target.append(target[tindice])  # assert

            work_image_paths[session].append(work_frames)
            work_targets[session].append(work_target)

            # Retrive the long history Information;
            if cfg.MODEL.LONG_MEMORY_ENABLE: # long history enbale; define as true; 
                
                assert cfg.MODEL.LONG_MEMORY_NUM_SAMPLES > 0

                long_start, long_end = (
                    max(0, work_start - cfg.MODEL.LONG_MEMORY_NUM_SAMPLES),
                    work_start - 1,
                )

                long_indices, long_key_padding_mask = sequence_sampler(
                    cfg,
                    long_start, 
                    long_end, 
                    mode,
                )
                
                long_indices = long_indices.clip(0)
                uindices, repeat_times = np.unique(
                    long_indices, return_counts=True
                )

                long_frames = []
                for indice in uindices:
                    for frame in chunk_indices[indice]:
                        if cfg.DATA.PATH_PREFIX == "hdd":
                            frame += 1
                        fname = cfg.DATA.FRAME_TEMPL.format(frame)
                        path = os.path.join(video_path, fname)
                        long_frames.append(path)

                long_image_paths[session].append(
                        [
                            long_frames,
                            repeat_times,
                            long_key_padding_mask,
                        ]
                    )
            else:
                
                long_image_paths[session].append(None)


    try:
        assert (
            len(
                set(
                    [
                        len(long_image_paths),
                        len(work_image_paths),
                        len(work_targets),
                    ]
                )
            )
            == 1
        )
    except:
        pdb.set_trace()

    if return_list:
        keys = work_image_paths.keys()

        work_image_paths = list(
            chain.from_iterable([work_image_paths[key] for key in keys])
        )  # 4836 个样本;

        long_image_paths = list(
            chain.from_iterable([long_image_paths[key] for key in keys])
        )  # 4836 个样本;

        work_targets = list(
            chain.from_iterable([work_targets[key] for key in keys])
        )
        assert (
            len(
                {
                    len(work_image_paths),
                    len(long_image_paths),
                    len(work_targets), 
                }
            )
            == 1
        )
        return (
            work_image_paths,
            long_image_paths,
            work_targets,
        )  

    return dict(work_image_paths), dict(long_image_paths), dict(work_targets)


def load_image_lists_batch_inference(
    cfg, sessions, video_root, target_root, mode, return_list=False
):


    work_image_paths = defaultdict(list)
    long_image_paths = defaultdict(list)
    work_targets = defaultdict(list)
    
    _work_indices = defaultdict(list) 
    _work_sessions = defaultdict(list)
    _work_frames = defaultdict(list)

    assert mode == 'test' 
    window_stride = 1 

    for session in sessions:
        
        video_path = os.path.join(video_root, session)
        target_path = os.path.join(target_root, session + ".npy")
        target = np.load(target_path)

        frame_list = sorted(
            os.listdir(video_path), key=lambda x: x[4:-4]
        ) 
        frame_indices = np.arange(len(frame_list))
        frame_length = len(frame_list)
        ## Split the Video List into chunks;
        num_chunks = int(frame_length // cfg.MODEL.CHUNK_SIZE)
        assert num_chunks == target.shape[0]

        if cfg.MODEL.CHUNK_SIZE > 1:
            chunk_indices = np.split(
                frame_indices[: num_chunks * cfg.MODEL.CHUNK_SIZE],
                num_chunks,
                axis=0,
            )
            chunk_indices = np.stack(chunk_indices, axis=0)

            ## Sample in each video chunks;
            ## Aplly Uniformly Sampling here；
            chunk_indices = chunk_indices[
                :, cfg.MODEL.CHUNK_SAMPLE_RATE // 2 :: cfg.MODEL.CHUNK_SAMPLE_RATE
            ]
        else:
            chunk_indices = frame_indices[:,None]

        seed = 0
        for work_start, work_end in zip(
            range(seed, num_chunks + 1, window_stride),
            range(
                seed + cfg.MODEL.WORK_MEMORY_NUM_SAMPLES,
                num_chunks + 1,
                window_stride, 
            ),
        ):  
            
            work_indices = np.arange(work_start, work_end).clip(0)[
                :: cfg.MODEL.WORK_MEMORY_SAMPLE_RATE
            ]  

            work_frames = []
            work_target = []

            for indice in work_indices: 
                
                findice = indice
                tindice = indice # 0

                for frame in chunk_indices[findice]: 
                    if cfg.DATA.PATH_PREFIX == "hdd":
                        frame += 1
                    fname = cfg.DATA.FRAME_TEMPL.format(frame)
                    path = os.path.join(video_path, fname)
                    work_frames.append(path)

                work_target.append(target[tindice])  

            work_image_paths[session].append(work_frames)  
            work_targets[session].append(work_target)

            _work_sessions[session].append(session) 
            _work_indices[session].append(work_indices)
            _work_frames[session].append(num_chunks)
            
            # Retrive the long history Information;
            if cfg.MODEL.LONG_MEMORY_ENABLE:
                assert cfg.MODEL.LONG_MEMORY_NUM_SAMPLES > 0
                
                long_start, long_end = (
                    max(0, work_start - cfg.MODEL.LONG_MEMORY_NUM_SAMPLES),
                    work_start - 1,
                )

                long_indices, long_key_padding_mask = sequence_sampler(
                    cfg,
                    long_start,
                    long_end,
                    mode,
                )

                long_indices = long_indices.clip(0)
                uindices, repeat_times = np.unique(
                    long_indices, return_counts=True
                )

                long_frames = []
                for indice in uindices:
                    for frame in chunk_indices[indice]:
                        if cfg.DATA.PATH_PREFIX == "hdd":
                            frame += 1
                        fname = cfg.DATA.FRAME_TEMPL.format(frame)
                        path = os.path.join(video_path, fname)
                        long_frames.append(path)

                long_image_paths[session].append(
                        [
                            long_frames, 
                            repeat_times, 
                            long_key_padding_mask, 
                        ]
                    )

            else:
                long_image_paths[session].append(None)  

    try:
        assert (
            len(
                set(
                    [
                        len(long_image_paths),
                        len(work_image_paths),
                        len(work_targets),
                        len(_work_indices),
                        len(_work_sessions),
                        len(_work_frames),
                    ]
                )
            )
            == 1
        )
    except:
        pdb.set_trace()

    if return_list:
        keys = work_image_paths.keys()

        work_image_paths = list(
            chain.from_iterable([work_image_paths[key] for key in keys])
        )  

        long_image_paths = list(
            chain.from_iterable([long_image_paths[key] for key in keys])
        )  

        work_targets = list(
            chain.from_iterable([work_targets[key] for key in keys])
        )

        _work_sessions = list(
            chain.from_iterable([_work_sessions[key] for key in keys])
        )

        _work_indices = list(
            chain.from_iterable([_work_indices[key] for key in keys])
        )
        
        _work_frames = list(
            chain.from_iterable([_work_frames[key] for key in keys])
        )
        
        assert (
            len(
                {
                    len(work_image_paths),
                    len(long_image_paths),
                    len(work_targets),
                    len(_work_indices), 
                    len(_work_sessions), 
                    len(_work_frames), 
                }
            )
            == 1
        )
        
        return (
            work_image_paths,
            long_image_paths,
            work_targets,
            _work_indices,
            _work_sessions,
            _work_frames,
        )  

    return dict(work_image_paths), dict(long_image_paths), dict(work_targets), dict(_work_indices), dict(_work_sessions), dict(_work_frames)


def load_frames(
    image_paths,
    max_spatial_scale=0,
    time_diff_prob=0.0,  
    gaussian_prob=0.0,
):
    num_retries = 10  # try times of the frame loading;

    augment_vid = (
        gaussian_prob > 0.0 or time_diff_prob > 0.0
    )  

    frames = utils.retry_load_images(
        image_paths,
        num_retries,
    )  # shape: torch.Size([64, 320, 400, 3])

    time_diff_aug = None
    if augment_vid:

        frames = frames.clone()
        (
            frames,
            time_diff_aug,
        ) = transform.augment_raw_frames( 
            frames, time_diff_prob, gaussian_prob
        )  

    return frames, time_diff_aug
