#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch

from collections import OrderedDict

import numpy as np
from sklearn.metrics import average_precision_score


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


def calibrated_average_precision_score(y_true, y_score):
    """Compute calibrated average precision (cAP), which is particularly
    proposed for the TVSeries dataset.
    """
    y_true_sorted = y_true[np.argsort(-y_score)]
    tp = y_true_sorted.astype(float)
    fp = np.abs(y_true_sorted.astype(float) - 1)
    tps = np.cumsum(tp)
    fps = np.cumsum(fp)
    ratio = np.sum(tp == 0) / np.sum(tp)
    cprec = tps / (
        tps + fps / (ratio + np.finfo(float).eps) + np.finfo(float).eps
    )
    cap = np.sum(cprec[tp == 1]) / np.sum(tp)
    return cap


def perframe_average_precision(
    ground_truth, prediction, class_names, ignore_index, metrics, postprocessing
):
    """Compute (frame-level) average precision between ground truth and
    predictions data frames.
    """
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Build metrics
    if metrics == "AP":
        compute_score = (
            average_precision_score  # defined from the sklearn function;
        )
    elif metrics == "cAP":
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError("Unknown metrics: {}".format(metrics))

    # Ignore backgroud class
    ignore_index = set([0, ignore_index])

    # Compute average precision
    result["per_class_AP"] = OrderedDict()
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            if np.any(ground_truth[:, idx]):
                result["per_class_AP"][class_name] = compute_score(
                    ground_truth[:, idx], prediction[:, idx]
                )
    result["mean_AP"] = np.mean(list(result["per_class_AP"].values()))

    return result


def get_stage_pred_scores(gt_targets, pred_scores, perc_s, perc_e):
    starts = []
    ends = [] 
    stage_gt_targets = []
    stage_pred_scores = [] 
    
    for i in range(len(gt_targets)):
        if gt_targets[i] == 0:  
            stage_gt_targets.append(gt_targets[i])        
            stage_pred_scores.append(pred_scores[i]) 
        else:
            if i == 0 or gt_targets[i - 1] == 0:
                starts.append(i) 
            if i == len(gt_targets) - 1 or gt_targets[i + 1] == 0: 
                ends.append(i)
    if len(starts) != len(ends):
        raise ValueError("starts and ends cannot pair!")

    action_lens = [ends[i] - starts[i] for i in range(len(starts))] 
    stage_starts = [
        starts[i] + int(action_lens[i] * perc_s) for i in range(len(starts))
    ]
    stage_ends = [
        max(stage_starts[i] + 1, starts[i] + int(action_lens[i] * perc_e))
        for i in range(len(starts))
    ]

    for i in range(len(starts)):

        stage_gt_targets.extend(gt_targets[stage_starts[i] : stage_ends[i]])

        stage_pred_scores.extend(pred_scores[stage_starts[i] : stage_ends[i]])

    return np.array(stage_gt_targets), np.array(stage_pred_scores)


def perstage_average_precision(
    ground_truth,
    prediction,
    class_names, 
    ignore_index,
    metrics,
    postprocessing,
):  
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Build metrics
    if metrics == "AP":
        compute_score = average_precision_score
    elif metrics == "cAP":
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError("Unknown metrics: {}".format(metrics))

    # Ignore backgroud class
    ignore_index = set([0, ignore_index])

    # Compute average precision
    for perc_s in range(10):
        perc_e = perc_s + 1
        stage_name = "{:2}%_{:3}%".format(
            perc_s * 10, perc_e * 10
        ) 
        result[stage_name] = OrderedDict(
            {"per_class_AP": OrderedDict()}
        )  # dicts;
        for idx, class_name in enumerate(class_names): 
            if idx not in ignore_index:
                stage_gt_targets, stage_pred_scores = get_stage_pred_scores(
                    (ground_truth[:, idx] == 1).astype(int),  #
                    prediction[:, idx], 
                    perc_s / 10,
                    perc_e / 10, 
                )
                result[stage_name]["per_class_AP"][class_name] = compute_score(
                    stage_gt_targets, stage_pred_scores
                )
        result[stage_name]["mean_AP"] = np.mean(
            list(result[stage_name]["per_class_AP"].values())
        )

    return result
