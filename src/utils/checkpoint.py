#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions that handle saving and loading of checkpoints."""

import pdb

import copy
import math
import numpy as np
import os
import pickle
from collections import OrderedDict
import torch

import src.utils.distributed as du
import src.utils.logging as logging
from src.utils.c2_model_loading import get_name_convert_func
from src.utils.env import checkpoint_pathmgr as pathmgr

logger = logging.get_logger(__name__)


def make_checkpoint_dir(path_to_job):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = os.path.join(path_to_job, "checkpoints")
    # Create the checkpoint dir from the master process
    if du.is_master_proc() and not pathmgr.exists(checkpoint_dir):
        try:
            pathmgr.mkdirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir


def get_checkpoint_dir(path_to_job):
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "checkpoints")


def get_path_to_checkpoint(path_to_job, epoch, task=""):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    if task != "":
        name = "{}_checkpoint_epoch_{:05d}.pyth".format(task, epoch)
    else:
        name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_job), name)


def get_last_checkpoint(path_to_job, task):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """

    d = get_checkpoint_dir(path_to_job)
    names = pathmgr.ls(d) if pathmgr.exists(d) else []
    if task != "":
        names = [f for f in names if "{}_checkpoint".format(task) in f]
    else:
        names = [f for f in names if f.startswith("checkpoint")]
    if len(names) == 0:
        return None
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)


def has_checkpoint(path_to_job):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job)
    files = pathmgr.ls(d) if pathmgr.exists(d) else []
    return any("checkpoint" in f for f in files)


def is_checkpoint_epoch(cfg, cur_epoch, multigrid_schedule=None):
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cfg (CfgNode): configs to save.
        cur_epoch (int): current number of epoch of the model.
        multigrid_schedule (List): schedule for multigrid training.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True
    if multigrid_schedule is not None:
        prev_epoch = 0
        for s in multigrid_schedule:
            if cur_epoch < s[-1]:
                period = max(
                    (s[-1] - prev_epoch) // cfg.MULTIGRID.EVAL_FREQ + 1, 1
                )
                return (s[-1] - 1 - cur_epoch) % period == 0
            prev_epoch = s[-1]

    return (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0


def save_checkpoint(path_to_job, model, optimizer, epoch, cfg, scaler=None):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
        scaler (GradScaler): the mixed precision scale.
    """
    # Save checkpoints only from the master process.
    if not du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        return
    # Ensure that the checkpoint dir exists.
    pathmgr.mkdirs(get_checkpoint_dir(path_to_job))
    # Omit the DDP wrapper in the multi-gpu setting.
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    normalized_sd = sub_to_normal_bn(sd)

    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": normalized_sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    # Write the checkpoint.
    path_to_checkpoint = get_path_to_checkpoint(
        path_to_job, epoch + 1, cfg.TASK
    )
    with pathmgr.open(path_to_checkpoint, "wb") as f:
        torch.save(checkpoint, f)
    return path_to_checkpoint


def inflate_weight(state_dict_2d, state_dict_3d):
    """
    Inflate 2D model weights in state_dict_2d to the 3D model weights in
    state_dict_3d. The details can be found in:
    Joao Carreira, and Andrew Zisserman.
    "Quo vadis, action recognition? a new model and the kinetics dataset."
    Args:
        state_dict_2d (OrderedDict): a dict of parameters from a 2D model.
        state_dict_3d (OrderedDict): a dict of parameters from a 3D model.
    Returns:
        state_dict_inflated (OrderedDict): a dict of inflated parameters.
    """
    state_dict_inflated = OrderedDict()
    for k, v2d in state_dict_2d.items():
        assert k in state_dict_3d.keys()
        v3d = state_dict_3d[k]
        # Inflate the weight of 2D conv to 3D conv.
        if len(v2d.shape) == 4 and len(v3d.shape) == 5:
            logger.info(
                "Inflate {}: {} -> {}: {}".format(k, v2d.shape, k, v3d.shape)
            )
            # Dimension need to be match.
            assert v2d.shape[-2:] == v3d.shape[-2:]
            assert v2d.shape[:2] == v3d.shape[:2]
            v3d = (
                v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
            )
        elif v2d.shape == v3d.shape:
            v3d = v2d
        else:
            logger.info(
                "Unexpected {}: {} -|> {}: {}".format(
                    k, v2d.shape, k, v3d.shape
                )
            )
        state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated


def load_checkpoint(
    cfg,
    path_to_checkpoint,
    model,
    data_parallel=True,
    optimizer=None,
    scaler=None,
    inflation=False,
    convert_from_caffe2=False,
    epoch_reset=False,
    clear_name_pattern=(),
    image_init=False,
    pretrained=False,
):
    
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        scaler (GradScaler): GradScaler to load the mixed precision scale.
        inflation (bool): if True, inflate the weights from the checkpoint.
        convert_from_caffe2 (bool): if True, load the model from caffe2 and
            convert it to pytorch.
        epoch_reset (bool): if True, reset #train iterations from the checkpoint.
        clear_name_pattern (string): if given, this (sub)string will be cleared
            from a layer name if it can be matched.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    logger.info("Loading network weights from {}.".format(path_to_checkpoint))

    # Account for the DDP wrapper in the multi-gpu setting.
    ms = model.module if data_parallel else model

    # Load the checkpoint on CPU to avoid GPU mem spike.
    with pathmgr.open(path_to_checkpoint, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")  # load from the memory
    model_state_dict_3d = (
        model.module.state_dict() if data_parallel else model.state_dict()
    )
    checkpoint["model_state"] = normal_to_sub_bn(  # ?
        checkpoint["model_state"], model_state_dict_3d
    )  
    new_trained_model = {}
    if pretrained:
        new_trained_model["cls_token"] = checkpoint["model_state"]["cls_token"]
        new_trained_model["patch_embed.proj.weight"] = checkpoint[
            "model_state"
        ]["patch_embed.proj.weight"]
        new_trained_model["patch_embed.proj.bias"] = checkpoint["model_state"][
            "patch_embed.proj.bias"
        ]
        new_trained_model["norm.weight"] = checkpoint["model_state"][
            "norm.weight"
        ]  # loading and enabling it;
        new_trained_model["norm.bias"] = checkpoint["model_state"]["norm.bias"]
        
        spatial_depth = cfg.MVIT.SPATIAL.DEPTH
        for k, v in model_state_dict_3d.items():
            # loading the spatial models here
            if "spatial_mvit" in k:
                k_re = k.split("spatial_mvit.")[-1]
                if "pool" in k:
                    new_trained_model[k] = checkpoint["model_state"][k_re].mean(
                        -3
                    )  # mean for temporal
                else:
                    new_trained_model[k] = checkpoint["model_state"][k_re]
            elif "temporal_mvit" in k:
                if 'fusion_block' in k:
                    continue 

                k_re = k.split("temporal_mvit.")[-1]
                k_re = k_re.split('.')[0] +'.' + str( int(k_re.split('.')[1]) + spatial_depth) + "." + '.'.join(k_re.split('.')[2:])

                if "pool" in k:
                    new_trained_model[k] = checkpoint["model_state"][k_re].mean(
                        -3, keepdim=True
                    )
                else:
                    new_trained_model[k] = checkpoint["model_state"][k_re]

            elif "temporal_compressor" in k:
                if cfg.MVIT.FUSION.EARLY_FUSION_TYPE=='kv':
                    continue
                k_re = k.split("temporal_compressor.")[-1]
                k_re = k_re.split('.')[0] +'.' + str( int(k_re.split('.')[1]) + cfg.MVIT.COMPRESSOR.START_LAYER) + "." + '.'.join(k_re.split('.')[2:])
                new_trained_model[k] = checkpoint["model_state"][k_re]
            pre_train_dict = new_trained_model  
    else:
        pre_train_dict = checkpoint["model_state"]
    
    # Match pre-trained weights that have same shape as current model.
    pre_train_dict_match = {}
    not_used_layers = []
    model_dict = ms.state_dict() 
    for k, v in pre_train_dict.items():
        if k in model_dict:

            if v.size() == model_dict[k].size():
                pre_train_dict_match[k] = v 
            else: 
                if "attn.rel_pos" in k:
                    v_shape = v.shape
                    v = v.t().unsqueeze(0)
                    v = torch.nn.functional.interpolate(
                        v,
                        size=model_dict[k].size()[0],
                        mode="linear",
                    )
                    v = v[0].t()
                    pre_train_dict_match[k] = v
                    logger.info(
                        "{} reshaped from {} to {}".format(
                            k, v_shape, v.shape
                        )
                    )
                elif "pos_embed_temporal" in k:
                    v_shape = v.shape
                    v = torch.nn.functional.interpolate(
                        v.permute(0, 2, 1),
                        size=model_dict[k].shape[1],
                        mode="linear",
                    )
                    pre_train_dict_match[k] = v.permute(0, 2, 1)
                    logger.info(
                        "{} reshaped from {} to {}".format(
                            k, v_shape, pre_train_dict_match[k].shape
                        )
                    )
                elif "pos_embed_spatial" in k:
                    v_shape = v.shape
                    pretrain_size = int(math.sqrt(v_shape[1]))
                    model_size = int(math.sqrt(model_dict[k].shape[1]))
                    assert pretrain_size * pretrain_size == v_shape[1]
                    assert model_size * model_size == model_dict[k].shape[1]
                    v = torch.nn.functional.interpolate(
                        v.reshape(
                            1, pretrain_size, pretrain_size, -1
                        ).permute(0, 3, 1, 2),
                        size=(model_size, model_size),
                        mode="bicubic",
                    )
                    pre_train_dict_match[k] = v.reshape(
                        1, -1, model_size * model_size
                    ).permute(0, 2, 1)
                    logger.info(
                        "{} reshaped from {} to {}".format(
                            k, v_shape, pre_train_dict_match[k].shape
                        )
                    )
                else:
                    not_used_layers.append(k)
        else:
            not_used_layers.append(k)
    # Weights that do not have match from the pre-trained model.
    not_load_layers = [
        k for k in model_dict.keys() if k not in pre_train_dict_match.keys()
    ]

    # Log weights that are not loaded with the pre-trained weights.
    if not_load_layers:
        for k in not_load_layers:
            logger.info("Network weights {} not loaded.".format(k))
    if not_used_layers:
        for k in not_used_layers:
            logger.info("Network weights {} not used.".format(k))
    # Load pre-trained weights.
    missing_keys, unexpected_keys = ms.load_state_dict(
        pre_train_dict_match, strict=False
    )  

    print("missing keys: {}".format(missing_keys))
    print("unexpected keys: {}".format(unexpected_keys))
    epoch = -1

    if "epoch" in checkpoint.keys() and not epoch_reset:
        epoch = checkpoint["epoch"]
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler_state"])
    else:
        epoch = -1
    return epoch


def sub_to_normal_bn(sd):
    """
    Convert the Sub-BN paprameters to normal BN parameters in a state dict.
    There are two copies of BN layers in a Sub-BN implementation: `bn.bn` and
    `bn.split_bn`. `bn.split_bn` is used during training and
    "compute_precise_bn". Before saving or evaluation, its stats are copied to
    `bn.bn`. We rename `bn.bn` to `bn` and store it to be consistent with normal
    BN layers.
    Args:
        sd (OrderedDict): a dict of parameters whitch might contain Sub-BN
        parameters.
    Returns:
        new_sd (OrderedDict): a dict with Sub-BN parameters reshaped to
        normal parameters.
    """
    new_sd = copy.deepcopy(sd)
    modifications = [
        ("bn.bn.running_mean", "bn.running_mean"),
        ("bn.bn.running_var", "bn.running_var"),
        ("bn.split_bn.num_batches_tracked", "bn.num_batches_tracked"),
    ]
    to_remove = ["bn.bn.", ".split_bn."]
    for key in sd:
        for before, after in modifications:
            if key.endswith(before):
                new_key = key.split(before)[0] + after
                new_sd[new_key] = new_sd.pop(key)

        for rm in to_remove:
            if rm in key and key in new_sd:
                del new_sd[key]

    for key in new_sd:
        if key.endswith("bn.weight") or key.endswith("bn.bias"):
            if len(new_sd[key].size()) == 4:
                assert all(d == 1 for d in new_sd[key].size()[1:])
                new_sd[key] = new_sd[key][:, 0, 0, 0]

    return new_sd


def c2_normal_to_sub_bn(key, model_keys):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        key (OrderedDict): source dict of parameters.
        mdoel_key (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    if "bn.running_" in key:
        if key in model_keys:
            return key

        new_key = key.replace("bn.running_", "bn.split_bn.running_")
        if new_key in model_keys:
            return new_key
    else:
        return key


def normal_to_sub_bn(checkpoint_sd, model_sd):
    """
    Convert BN parameters to Sub-BN parameters if model contains Sub-BNs.
    Args:
        checkpoint_sd (OrderedDict): source dict of parameters.
        model_sd (OrderedDict): target dict of parameters.
    Returns:
        new_sd (OrderedDict): converted dict of parameters.
    """
    for key in model_sd:
        if key not in checkpoint_sd:
            if "bn.split_bn." in key:
                load_key = key.replace("bn.split_bn.", "bn.")
                bn_key = key.replace("bn.split_bn.", "bn.bn.")
                checkpoint_sd[key] = checkpoint_sd.pop(load_key)
                checkpoint_sd[bn_key] = checkpoint_sd[key]

    for key in model_sd:
        if key in checkpoint_sd:
            model_blob_shape = model_sd[key].shape
            c2_blob_shape = checkpoint_sd[key].shape

            if (
                len(model_blob_shape) == 1
                and len(c2_blob_shape) == 1
                and model_blob_shape[0] > c2_blob_shape[0]
                and model_blob_shape[0] % c2_blob_shape[0] == 0
            ):
                before_shape = checkpoint_sd[key].shape
                checkpoint_sd[key] = torch.cat(
                    [checkpoint_sd[key]]
                    * (model_blob_shape[0] // c2_blob_shape[0])
                )
                logger.info(
                    "{} {} -> {}".format(
                        key, before_shape, checkpoint_sd[key].shape
                    )
                )
    return checkpoint_sd


def load_test_checkpoint(cfg, model):
    """
    Loading checkpoint logic for testing.
    """
    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in MODEL_VIS.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TEST.CHECKPOINT_FILE_PATH and test it.
        load_checkpoint(
            cfg = cfg, 
            path_to_checkpoint = cfg.TEST.CHECKPOINT_FILE_PATH,
            model = model,
            data_parallel=cfg.NUM_GPUS > 1,
            optimizer=None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
            pretrained=False, 
            epoch_reset=True, 
        )
    elif has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR, cfg.TASK)
        load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        logger.info(
            "Unknown way of loading checkpoint. Using with random initialization, only for debugging."
        )


def load_train_checkpoint(cfg, model, optimizer, scaler=None):
    """
    Loading checkpoint logic for training.
    """
    if cfg.TRAIN.AUTO_RESUME and has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR, cfg.TASK)
        logger.info("Load from last checkpoint, {}.".format(last_checkpoint))
        checkpoint_epoch = load_checkpoint(
            last_checkpoint,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler=scaler,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler=scaler,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    return start_epoch
