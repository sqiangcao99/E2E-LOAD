#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import pdb
import numpy as np
import os
import pickle
import json
import torch

import src.utils.checkpoint as cu
import src.utils.distributed as du
import src.utils.logging as logging
import src.utils.misc as misc
import src.visualization.tensorboard_vis as tb
from src.datasets import loader
from src.models import build_model
from src.utils.env import pathmgr
from src.utils.meters import TestMeter

logger = logging.get_logger(__name__)

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
         src/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, items in enumerate(test_loader): 

        if cfg.MODEL.LONG_MEMORY_ENABLE:
            (work_frames, long_frames, long_key_padding_mask, labels, session, indice, num_frame    ) = items
        else:
            (
                work_frames,
                labels, session, indice, num_frame
            ) = items  # w_f: torch.Size([2, 3, 64, 224, 224]); label: torch.Size([2, 32, 31])

        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(work_frames, (list,)):
                for i in range(len(work_frames)):
                    work_frames[i] = work_frames[i].cuda(non_blocking=True)
            else:
                work_frames = work_frames.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            ## Process And Remember the current frames; 

            if cfg.MODEL.LONG_MEMORY_ENABLE:

                if isinstance(long_frames, (list,)):
                    for i in range(len(long_frames)):
                        long_frames[i] = long_frames[i].cuda(non_blocking=True)
                else:
                    long_frames = long_frames.cuda(non_blocking=True)

        test_meter.data_toc() 
        if cfg.MODEL.LONG_MEMORY_ENABLE:
            preds = model(work_frames, long_frames,long_key_padding_mask)
        else:
            preds = model(work_frames)

        pdb.set_trace() 
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds = du.all_gather([preds]) 
        if cfg.NUM_GPUS:  
            preds = preds.cpu()
            labels = labels.cpu()
            
        test_meter.iter_toc() 
        test_meter.update_stats(preds.detach(), labels, session, indice, num_frame)
        
        test_meter.log_iter_stats(cur_iter) 
        test_meter.iter_tic() 
 
    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
         src/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0: 
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:  

        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view 

        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)
        # Build the video model and print model statistics.
        model = build_model(cfg)
    
        flops, params = 0.0, 0.0
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            model.eval()
            flops, params = misc.log_model_info( # TODO: Fix cal the parameters for this model; 
                model, cfg, use_train_input=False
            )

        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)
        
        cu.load_test_checkpoint(cfg, model)
        # Create video testing loaders.
        test_loader = loader.construct_loader(cfg, "test")  

        logger.info("Testing model for {} iterations".format(len(test_loader)))
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
  
        test_meter = TestMeter(
            cfg = cfg,
            num_videos = test_loader.dataset.num_videos,  
            num_clips=test_loader.dataset.num_clips,
            num_cls = cfg.MODEL.NUM_CLASSES,
            overall_iters = len(test_loader),
            multi_label=cfg.DATA.MULTI_LABEL,
            ensemble_method = cfg.DATA.ENSEMBLE_METHOD,
        )
        # # Perform multi-view test on the entire dataset.
        test_meter = perform_test(
            test_loader, model, test_meter, cfg
        )  

        ## Save the results: 
        if du.is_master_proc():
            pickle.dump({
                'cfg': cfg,
                'perframe_pred_scores': test_meter.video_preds,
                'perframe_gt_targets': test_meter.video_labels,
            }, open(os.path.splitext(cfg.TEST.CHECKPOINT_FILE_PATH)[0] + '.pkl', 'wb'))
            
            if cfg.LOG_MODEL_INFO:
                results = test_meter.results
                logger.info('Action detection perframe m{}: {:.5f}'.format(
                    cfg.DATA.METRICS, results['mean_AP']
                )) 

