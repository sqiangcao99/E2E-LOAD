#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Learning rate policy."""

import pdb

import math


def get_lr_at_epoch(cfg, cur_epoch): 
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = get_lr_func(cfg.SOLVER.LR_POLICY)(cfg, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
        lr_start = cfg.SOLVER.WARMUP_START_LR
        lr_end = get_lr_func(cfg.SOLVER.LR_POLICY)(
            cfg, cfg.SOLVER.WARMUP_EPOCHS
        )
        alpha = (lr_end - lr_start) / cfg.SOLVER.WARMUP_EPOCHS
        lr = cur_epoch * alpha + lr_start
    return lr

def lr_func_cosine(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    offset = cfg.SOLVER.WARMUP_EPOCHS if cfg.SOLVER.COSINE_AFTER_WARMUP else 0.0
    assert cfg.SOLVER.COSINE_END_LR < cfg.SOLVER.BASE_LR  #

    return (
        cfg.SOLVER.COSINE_END_LR
        + (cfg.SOLVER.BASE_LR - cfg.SOLVER.COSINE_END_LR)
        * (
            math.cos(
                math.pi * (cur_epoch - offset) / (cfg.SOLVER.MAX_EPOCH - offset)
            )
            + 1.0
        )
        * 0.5
    )

def lr_func_steps_with_relative_lrs(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    steps with relative learning rate schedule.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    ind = get_step_index(cfg, cur_epoch)
    return cfg.SOLVER.LRS[ind] * cfg.SOLVER.BASE_LR


def get_step_index(cfg, cur_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg.SOLVER.STEPS + [cfg.SOLVER.MAX_EPOCH]
    for ind, step in enumerate(steps): 
        if cur_epoch < step:
            break
    return ind - 1


def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy

    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]


def retrieve_grouped_lr(model): 

    opt_parameters = (
        []
    )  # To be passed to the optimizer (only parameters of the layers you want to update).

    named_parameters = list(model.named_parameters())  #

    pdb.set_trace()

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
    set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]

    init_lr = 1e-6  #

    for i, (name, params) in enumerate(named_parameters):

        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01

        if name.startswith("roberta_model.embeddings") or name.startswith(
            "roberta_model.encoder"
        ):
            # For first set, set lr to 1e-6 (i.e. 0.000001)
            lr = init_lr

            # For set_2, increase lr to 0.00000175
            lr = init_lr * 1.75 if any(p in name for p in set_2) else lr

            # For set_3, increase lr to 0.0000035
            lr = (
                init_lr * 3.5 if any(p in name for p in set_3) else lr
            ) 

            opt_parameters.append(
                {"params": params, "weight_decay": weight_decay, "lr": lr}
            )

        # For regressor and pooler, set lr to 0.0000036 (slightly higher than the top layer).
        if name.startswith("regressor") or name.startswith(
            "roberta_model.pooler"
        ):
            lr = init_lr * 3.6

            opt_parameters.append(
                {"params": params, "weight_decay": weight_decay, "lr": lr}
            )

    return opt_parameters
