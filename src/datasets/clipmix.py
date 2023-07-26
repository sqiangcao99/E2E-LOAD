# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py,
published under an Apache License 2.0.

COMMENT FROM ORIGINAL:
Mixup and Cutmix
Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899) # NOQA
Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""

import pdb

import einops
import numpy as np
import torch

def rand_bbox(num_samples, lam, margin=0.0, count=None):
    """
    Generates a random square bbox based on lambda value.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    
    ratio = np.sqrt(1 - lam)
    cut_l = int(num_samples * ratio)
    margin_l = int(margin * cut_l)

    cl = np.random.randint(0 + margin_l, num_samples - margin_l, size=count)
    st = np.clip(cl- cut_l // 2, 0, num_samples)
    et = np.clip(cl + cut_l // 2, 0, num_samples)
    
    return st, et


def get_cutmix_bbox(num_samples, lam, correct_lam=True, count=None):
    """
    Generates the box coordinates for cutmix.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        correct_lam (bool): Apply lambda correction when cutmix bbox clipped by
            image borders.
        count (int): Number of bbox to generate
    """

    (st, et) = rand_bbox(num_samples, lam, count=count)
    if correct_lam:
        bbox_area = et - st
        lam = 1.0 - bbox_area / float(num_samples)
    return (st, et), lam


class ClipMix:
    """
    Apply mixup and/or cutmix for videos at batch level.
    mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
        Features (https://arxiv.org/abs/1905.04899)
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        mix_prob=1.0,
        num_samples=32, 
        correct_lam=True,
    ):
        """
        Args:
            mixup_alpha (float): Mixup alpha value.
            cutmix_alpha (float): Cutmix alpha value.
            mix_prob (float): Probability of applying mixup or cutmix.
            switch_prob (float): Probability of switching to cutmix instead of
                mixup when both are active.
            correct_lam (bool): Apply lambda correction when cutmix bbox
                clipped by image borders.
            label_smoothing (float): Apply label smoothing to the mixed target
                tensor. If label_smoothing is not used, set it to 0.
            num_classes (int): Number of classes for target.
        """
        self.mixup_alpha = mixup_alpha # 0.8
        self.mix_prob = mix_prob # 1.0
        self.correct_lam = correct_lam 
        self.num_samples = num_samples

    def _get_mixup_params(self):
        lam = 1.0
        use_clipmix = False 

        if np.random.rand() < self.mix_prob: # don't apply it;  
            lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            use_clipmix = True
            lam = float(lam_mix) 
        return lam, use_clipmix

    def _mix_batch(self, x, target):

        lam, use_clipmix = self._get_mixup_params() # torch.Size([2, 3, 48, 224, 224]) all images in single batch; 
        # pdb.set_trace() 
        
        if use_clipmix:
            (st, et), lam = get_cutmix_bbox(
                self.num_samples,
                lam, 
                correct_lam=self.correct_lam,
            )
            x = einops.rearrange(x, 'b c (t ck) h w -> b c t ck h w', t=self.num_samples)            
            x[:,:,st:et] = x.flip(0)[:,:,st:et]
            target[:, st:et] = target.flip(0)[:, st:et]
            x = einops.rearrange(x, 'b c t ck h w -> b c (t ck) h w', t=self.num_samples)   
            
        return x, target
        
    

    def __call__(self, x, target): # apply on the input data, and the targets; 
        
        x, target = self._mix_batch(x, target)
        # pdb.set_trace() 
        
        return x, target