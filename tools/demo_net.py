#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import pdb
import numpy as np
import os
import pickle
import json
import torch
import time
from datetime import datetime 

from bisect import bisect_right

import os.path as osp

import src.utils.checkpoint as cu
import src.utils.distributed as du
import src.utils.logging as logging
import src.utils.misc as misc
import src.visualization.tensorboard_vis as tb
from src.datasets import loader
from src.models import build_model
from src.utils.env import pathmgr
from src.utils.meters import TestMeter
import src.datasets.helper as helper
import src.datasets.utils as utils
import src.utils.evalution as evalution

logger = logging.get_logger(__name__)

def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
         src/config/defaults.py
    """
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Online Inference Speed Test.")
    # logger.info(cfg)
    model = build_model(cfg)
    flops, params = 0.0, 0.0
    
    # Setuo model to test mode. 
    model.eval()

    # params = torch.load(cfg.TEST.CHECKPOINT_FILE_PATH)
    num_params = sum(param.numel() for param in model.parameters())
    
    logger.info("Parames(M): {}".format(num_params/1e6))
    
    cu.load_test_checkpoint(cfg, model) 
    logger.info("Succeed Loading the PreTrained Parameters.")

    def to_device(x, dtype=np.float32):
        return torch.as_tensor(x.astype(dtype)).unsqueeze(0).to(device)

    if cfg.DEMO.ALL_TEST:
        sessions = getattr(cfg.DATA, "TEST_SESSION_SET")
    else:
        sessions = cfg.DEMO.INPUT_VIDEO

    num_video = len(sessions)
    
    inference_time = 0
    total_frames = 0

    pred_scores = {}
    gt_targets = {}
    
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = osp.splitext(cfg.TEST.CHECKPOINT_FILE_PATH)[0] + '_' + stamp
    
    # loading the video info; 
    for idx, session in enumerate(sessions):
        data_root = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.DATA.PATH_PREFIX)
        target_root = os.path.join(data_root, cfg.DATA.TARGET_FORDER)

        video_path = os.path.join(data_root, cfg.DATA.VIDEO_FORDER, session)
        target_path = os.path.join(target_root, session + ".npy")    
        
        # Load the related targets; 
        target = np.load(target_path)

        frame_length = len(os.listdir(video_path))
        frame_indices = np.arange(frame_length)
        
        num_chunks = int(frame_length // cfg.MODEL.CHUNK_SIZE)
        total_frames += num_chunks
        
        assert num_chunks == target.shape[0]
        
        chunk_indices = np.split(
            frame_indices[: num_chunks * cfg.MODEL.CHUNK_SIZE],
            num_chunks,
            axis=0,
        )
        
        
        chunk_indices = np.stack(chunk_indices, axis=0)
         
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
        indices_list = []

        single_pred = []
        single_gt = []

        model.empty_cache()
        
        with torch.no_grad():
            
            for work_start, work_end in zip(range(0, num_chunks + 1),
                                            range(cfg.MODEL.WORK_MEMORY_NUM_SAMPLES, num_chunks + 1)):  
                target = target[::cfg.MODEL.WORK_MEMORY_SAMPLE_RATE]
                work_indices = np.arange(work_start, work_end).clip(0)
                work_indices = work_indices[::cfg.MODEL.WORK_MEMORY_SAMPLE_RATE] 

                # Retrive the frames paths here; 
                work_frames = []
                work_target = []
                
                if work_start == 0: 
                    for indice in work_indices:     
                        for frame in chunk_indices[indice]: 
                            
                            if cfg.DATA.PATH_PREFIX == "hdd":
                                frame += 1
                            
                            fname = cfg.DATA.FRAME_TEMPL.format(frame)
                            path = os.path.join(video_path, fname)
                            work_frames.append(path)
                        work_target.append(target[indice]) 
                else:
                    indice = work_indices[-1]

                    for frame in chunk_indices[indice]: 
                        
                        if cfg.DATA.PATH_PREFIX == "hdd":
                            frame += 1
                        
                        fname = cfg.DATA.FRAME_TEMPL.format(frame)
                        path = os.path.join(video_path, fname)
                        work_frames.append(path)
                    work_target.append(target[indice]) 
                    
                long_end = work_start - 1 
                long_start = long_end - cfg.MODEL.LONG_MEMORY_NUM_SAMPLES * cfg.MODEL.LONG_MEMORY_SAMPLE_RATE
                long_indices = np.arange(long_start+1, long_end+1)
         
                ulong_indices, repeat_times = np.unique(long_indices, return_counts=True) 

                memory_key_padding_mask = np.zeros(len(long_indices))
                last_zero = bisect_right(long_indices, 0) - 1 
                if last_zero > 0:
                    memory_key_padding_mask[:last_zero] = float('-inf')
                memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32)).unsqueeze(0) 

                # Load the images;
                min_scale = cfg.DATA.TEST_CROP_SIZE
                max_scale = cfg.DATA.TEST_CROP_SIZE  
                crop_size = cfg.DATA.TEST_CROP_SIZE 

                
                work_frames, _ = helper.load_frames(
                    work_frames,
                    max_spatial_scale=min_scale,
                    time_diff_prob=0.0,
                )  # work_frames.shape: torch.Size([96, 180, 320, 3])

                work_frames = work_frames.float()
                work_frames = work_frames / 255.0

                # Perform color normalization.
                work_frames = utils.tensor_normalize(work_frames, cfg.DATA.MEAN, cfg.DATA.STD)
                work_frames = work_frames.permute(3, 0, 1, 2)

                work_frames = utils.spatial_sampling(
                    work_frames,
                    spatial_idx=1, 
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=False, 
                    inverse_uniform_sampling=False, 
                    aspect_ratio=None, 
                    scale=None, 
                    motion_shift=False, 
                ) # C, NF, H, W; 

                work_frames = work_frames.cuda(non_blocking=True).unsqueeze(0)
                # work_frames = work_frames.unsqueeze(0)
                
                start = time.time()
                score = model.stream_inference(work_frames, ulong_indices, repeat_times, memory_key_padding_mask)
                
                delta = time.time() - start
                score = score[0].softmax(dim=-1).cpu().numpy() # torch.Size([1, 32, 22])
                
                inference_time += delta

                if work_start == 0: 
                    single_pred.extend(list(score))
                    single_gt.extend(work_target)
                else:
                    single_pred.extend([list(score[-1])])
                    single_gt.extend(work_target)
        
        assert {len(single_pred), num_chunks, len(single_gt)}
        # performing the single test
        
        result = evalution.eval_perframe(
                cfg,
                single_gt,
                single_pred,
        )       
        
        logger.info('Process: {}/{}'.format(idx+1, num_video))
        logger.info('Video Info: name: {}, num_chunks: {} mAP: {}'.format(session, num_chunks, result["mean_AP"]))

        pred_scores[session] = single_pred
        gt_targets[session] = single_gt

    logger.info('Performing the Last Test.')
    results = evalution.eval_perframe(
            cfg,
            np.concatenate(list(gt_targets.values()), axis=0),
            np.concatenate(list(pred_scores.values()), axis=0),
    )

    logger.info("All Inference Time, {}".format(inference_time))
    logger.info("Num Chunks, {}".format(total_frames))
    logger.info("FPS, {}".format(float(total_frames/inference_time)))
    logger.info("mAP, {}".format(results["mean_AP"]))

    
    if cfg.DEMO.ALL_TEST:
        logger.info('Saving Predicted Files to: {}'.format(save_path))

        pickle.dump({
            'cfg': cfg,
            'perframe_pred_scores': pred_scores,
            'perframe_gt_targets': gt_targets,
        }, open(save_path + '.pkl', 'wb'))
