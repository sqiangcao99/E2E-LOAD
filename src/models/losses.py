#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import pdb

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchvideo.losses.soft_target_cross_entropy import (
    SoftTargetCrossEntropyLoss,
)

class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, dummy_labels=None):
        targets = torch.zeros(inputs.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss(reduction=self.reduction).cuda()(
            inputs, targets
        )
        return loss


class MultipleMSELoss(nn.Module):
    """
    Compute multiple mse losses and return their average.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MultipleMSELoss, self).__init__()
        self.mse_func = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        loss_sum = 0.0
        multi_loss = []
        for xt, yt in zip(x, y):
            if isinstance(yt, (tuple,)):
                if len(yt) == 2:
                    yt, wt = yt
                    lt = "mse"
                elif len(yt) == 3:
                    yt, wt, lt = yt
                else:
                    raise NotImplementedError
            else:
                wt, lt = 1.0, "mse"
            if lt == "mse":
                loss = self.mse_func(xt, yt)
            else:
                raise NotImplementedError
            loss_sum += loss * wt
            multi_loss.append(loss)
        return loss_sum, multi_loss

class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(BinaryCrossEntropyLoss, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        self.ignore_index = ignore_index 

    def forward(self, input, target):

        if self.ignore_index >= 0:
            notice_index = [
                i for i in range(target.shape[-1]) if i != self.ignore_index
            ]
            
            return self.criterion(input[:, notice_index], target[:, notice_index])

        else:

            return self.criterion(input, target)

class MultipCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", ignore_index=-100):
        super(MultipCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            
            notice_index = [
                i for i in range(target.shape[-1]) if i != self.ignore_index
            ]
            
            output = torch.sum(
                -target[:, notice_index] * logsoftmax(input[:, notice_index]),
                dim=1,
            ) 

            if self.reduction == "mean":
                return torch.mean(output[target[:, self.ignore_index] != 1])
            elif self.reduction == "sum":
                return torch.sum(output[target[:, self.ignore_index] != 1])
            else:
                return output[target[:, self.ignore_index] != 1]
        else:
            
            output = torch.sum(-target * logsoftmax(input), dim=1)

            if self.reduction == "mean":
                
                return torch.mean(output)
            elif self.reduction == "sum":
                return torch.sum(output)
            else:
                return output


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    # "bce_logit": nn.BCEWithLogitsLoss, 
    "bce_logit": BinaryCrossEntropyLoss,
    "soft_cross_entropy": partial(
        SoftTargetCrossEntropyLoss, normalize_targets=False
    ),
    "contrastive_loss": ContrastiveLoss,
    "mse": nn.MSELoss,
    "multi_mse": MultipleMSELoss,
    "multi_ce": MultipCrossEntropyLoss,
}


        

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

