#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import pdb

import random
import math
import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import src.models.losses as losses
import src.models.optimizer as optim
import src.utils.checkpoint as cu
import src.utils.distributed as du
import src.utils.logging as logging
import src.utils.metrics as metrics
import src.utils.misc as misc
import src.visualization.tensorboard_vis as tb
from src.datasets import loader
from src.datasets.mixup import MixUp
from src.datasets.clipmix import ClipMix
from src.models import build_model
from src.models.contrastive import (
    contrastive_forward,
    contrastive_parameter_surgery,
)
from src.utils.meters import EpochTimer, TrainMeter, ValMeter
from src.utils.multigrid import MultigridSchedule

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
         src/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader) 
    # pdb.set_trace() 
    if cfg.MIXUP.ENABLE: 
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    if cfg.AUG.CURRENT.MIX.ENABLE:

        clipmix_fn = ClipMix(
            mixup_alpha=cfg.AUG.CURRENT.MIX.ALPHA,
            mix_prob=cfg.AUG.CURRENT.MIX.PROB,
            num_samples=cfg.MODEL.WORK_MEMORY_NUM_SAMPLES,
        )

    if cfg.MODEL.FROZEN_BN:
        misc.frozen_bn_stats(model)
    # Explicitly declare reduction to mean.
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(
        reduction="mean", ignore_index=cfg.DATA.IGNORE_INDEX
    )
    
    for cur_iter, items in enumerate(train_loader):
        if cfg.MODEL.LONG_MEMORY_ENABLE:
            (work_frames, long_frames, long_key_padding_mask, labels) = items
        else:
            (
                work_frames,
                labels,
            ) = items  # w_f: torch.Size([2, 3, 64, 224, 224]); label: torch.Size([2, 32, 31])

        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(work_frames, (list,)): 
                for i in range(len(work_frames)): 
                    if isinstance(work_frames[i], (list,)):
                        for j in range(len(work_frames[i])):
                            work_frames[i][j] = work_frames[i][j].cuda(
                                non_blocking=True
                            )
                            if cfg.MODEL.LONG_MEMORY_ENABLE:
                                long_frames[i][j] = long_frames[i][j].cuda(
                                    non_blocking=True
                                )
                                long_key_padding_mask[i][j] = long_key_padding_mask[i][j].cuda(non_blocking=True)
                    else:
                        work_frames[i] = work_frames[i].cuda(non_blocking=True)
                        if cfg.MODEL.LONG_MEMORY_ENABLE:
                            long_frames[i] = long_frames[i].cuda(
                                non_blocking=True
                            )
                            long_key_padding_mask[i][j] = long_key_padding_mask[i][j].cuda(non_blocking=True)
            else: 
                work_frames = work_frames.cuda(non_blocking=True)
                if cfg.MODEL.LONG_MEMORY_ENABLE:
                    long_frames = long_frames.cuda(non_blocking=True)
                    long_key_padding_mask = long_key_padding_mask.cuda(non_blocking=True)
            if not isinstance(labels, list):
                labels = labels.cuda(non_blocking=True)

        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size

        # retrive the
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        if cfg.MIXUP.ENABLE:
            work_frames, labels = mixup_fn(work_frames, labels)

        if cfg.AUG.CURRENT.MIX.ENABLE:
            work_frames, labels = clipmix_fn(work_frames, labels)


        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
           
            perform_backward = True
            optimizer.zero_grad()

            if cfg.MASK.ENABLE:
                preds, labels = model(inputs) # ? 
            elif cfg.MODEL.LONG_MEMORY_ENABLE:
                preds = model(work_frames, long_frames,long_key_padding_mask)
            else:
                preds = model(work_frames)
            
            # Backward the Loss;
            preds = preds.reshape(-1, cfg.DATA.NUM_CLASSES)
            labels = labels.reshape(-1, cfg.DATA.NUM_CLASSES)
            
            labels_num = labels.sum(-1, keepdims=True)
            labels_mask = labels>0

            if cfg.MODEL.MULTI_LABEL_SMOOTH:
                off_value = cfg.MODEL.SMOOTH_VALUE / cfg.MODEL.NUM_CLASSES 
                on_value = - cfg.MODEL.SMOOTH_VALUE + off_value 

                labels[labels_mask] += on_value
                labels[~labels_mask] += off_value

            if cfg.MODEL.MULTI_LABEL_NORM:
                labels = labels / labels_num
            loss = loss_fun(preds, labels)

        loss_extra = None
        if isinstance(loss, (list, tuple)):
            loss, loss_extra = loss 

        # check Nan Loss.
        misc.check_nan_losses(loss)

        if perform_backward:
            scaler.scale(loss).backward() 
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer) 
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            grad_norm = torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        else:
            grad_norm = optim.get_grad_norm_(model.parameters()) 

        # Update the parameters. (defaults to True)

        model, update_param = contrastive_parameter_surgery(
            model, cfg, epoch_exact, cur_iter
        ) 
        if update_param:
            scaler.step(optimizer)
        scaler.update()

        if cfg.DATA.MULTI_LABEL: 
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, grad_norm = du.all_reduce([loss, grad_norm])
            loss, grad_norm = (
                loss.item(),
                grad_norm.item(),
            )
        elif cfg.MASK.ENABLE:
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, grad_norm = du.all_reduce([loss, grad_norm])
                if loss_extra:
                    loss_extra = du.all_reduce(loss_extra)
            loss, grad_norm = (
                loss.item(),
                grad_norm.item(),
            )
            if loss_extra:
                loss_extra = [one_loss.item() for one_loss in loss_extra]

        # Update and log stats.
        train_meter.update_stats(
            loss,
            lr,
            grad_norm,
            preds.shape[0]
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            loss_extra,
        )
        # write to tensorboard format if available.
        if writer is not None:  ### ???
            writer.add_scalars(
                {
                    "Train/loss": loss,
                    "Train/lr": lr,
                },
                global_step=data_size * cur_epoch
                + cur_iter, 
            )
        train_meter.iter_toc()  # do measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()

    del work_frames 

    # in case of fragmented memory
    torch.cuda.empty_cache()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset() 


@torch.no_grad()
def eval_epoch(
    val_loader, model, val_meter, cur_epoch, cfg, train_loader, writer
):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
         src/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()

    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

    val_meter.iter_tic() 

    for cur_iter, items in enumerate(val_loader):

        if cfg.MODEL.LONG_MEMORY_ENABLE:
            (work_frames, long_frames, long_key_padding_mask, labels) = items
        else:
            (
                work_frames,
                labels,
            ) = items  # w_f: torch.Size([2, 3, 64, 224, 224]); label: torch.Size([2, 32, 31])
        # (inputs, labels, index, time, meta)
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(work_frames, (list,)):
                for i in range(len(work_frames)):
                    work_frames[i] = work_frames[i].cuda(non_blocking=True)

                    if cfg.MODEL.LONG_MEMORY_ENABLE:
                        long_frames[i] = long_frames[i].cuda(non_blocking=True)
            else:
                work_frames = work_frames.cuda(non_blocking=True)
                if cfg.MODEL.LONG_MEMORY_ENABLE:
                    long_frames = long_frames.cuda(non_blocking=True)
                    long_key_padding_mask = long_key_padding_mask.cuda(non_blocking=True)

            labels = labels.cuda()

        val_meter.data_toc()  # finish recording the data loading time;

        if cfg.MODEL.LONG_MEMORY_ENABLE:
            preds = model(work_frames, long_frames, long_key_padding_mask)
        else:
            preds = model(work_frames)
            

        preds = preds.reshape(-1, cfg.DATA.NUM_CLASSES)
        labels = labels.reshape(-1, cfg.DATA.NUM_CLASSES)
        
        val_loss = loss_fun(preds, labels) 

        if cfg.NUM_GPUS > 1:
            val_loss = du.all_reduce([val_loss]) 
            preds, labels = du.all_gather([preds, labels]) 

            val_loss = val_loss[0].item()
        else:
            val_loss = val_loss.item()

        val_meter.iter_toc()  # the data time of the single iterations(inference+dataloading)
        # Update and log stats.
        val_meter.update_stats( 
            val_loss,
            preds.shape[0]
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )
        # write to tensorboard format if available.

        if writer is not None:  # what's this for?
            writer.add_scalars(
                {"Val/Loss": loss},
                global_step=len(val_loader) * cur_epoch + cur_iter,
            )

        val_meter.update_predictions(preds, labels) 

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()  

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch) 
    # write to tensorboard format if available.
    if writer is not None:

        all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
        all_labels = [
            label.clone().detach() for label in val_meter.all_labels
        ] 

        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
        writer.plot_eval(  # ?
            preds=all_preds, labels=all_labels, global_step=cur_epoch
        )

    val_meter.reset() 


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)

def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
         src/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # setup_random_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    flops, params = 0.0, 0.0
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        flops, params = misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # pdb.set_trace()
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task=cfg.TASK)
        if last_checkpoint is not None:
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
            start_epoch = checkpoint_epoch + 1
        elif "ssl_eval" in cfg.TASK:
            last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task="ssl")
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
                epoch_reset=True,
                clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            )
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = 0
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "": 
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg,
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
            pretrained=True,
            # MayFix latter; 
        )
        start_epoch = checkpoint_epoch + 1 # 
    else:
        start_epoch = 0

    ## retrive the checkpoint save path;

    # Create the video train and val loaders.
    # pdb.set_trace() 
    train_loader = loader.construct_loader(cfg, "train") # here retrive the dataloader; 
    val_loader = loader.construct_loader(cfg, "val")
    
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    if (
        cfg.TASK == "ssl"  # ssl means self-supervised learning;
        and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.KNN_ON
    ):
        if hasattr(model, "module"):
            model.module.init_knn_labels(train_loader)
        else:
            model.init_knn_labels(train_loader)

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        
        # Shuffle the dataset.
        # loader.shuffle_dataset(train_loader, cur_epoch)
        if hasattr(train_loader.dataset, "_set_epoch_num"):
            train_loader.dataset._set_epoch_num(cur_epoch)  

        # Train for one epoch.
        epoch_timer.epoch_tic()        
        # debuging the eval code;

        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
            writer,
        )
        epoch_timer.epoch_toc()

        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        # reconstruct the training loader; 
        train_loader = loader.construct_loader(cfg, "train") 

        is_checkp_epoch = (
            cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
                None,
            )
            or cur_epoch == cfg.SOLVER.MAX_EPOCH - 1
        )
        is_eval_epoch = (
            misc.is_eval_epoch(
                cfg,
                cur_epoch,
                None,
            )
            and not cfg.MASK.ENABLE
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(
                val_loader,
                model,
                val_meter,
                cur_epoch,
                cfg,
                train_loader,
                writer,
            )



    if (
        start_epoch == cfg.SOLVER.MAX_EPOCH
    ):  # eval if we loaded the final checkpoint
        eval_epoch(
            val_loader, model, val_meter, start_epoch, cfg, train_loader, writer
        )

    if writer is not None:
        writer.close()

