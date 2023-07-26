# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

import pdb

import einops

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

import src.utils.logging as logging
import src.utils.weight_init_helper as init_helper

from src.models.modules.attention_spatial import MultiScaleBlock
from src.models.batchnorm_helper import get_norm
from src.models.common import TwoStreamFusion
from src.models.reversible_mvit import ReversibleMViT
from src.models.utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
    validate_checkpoint_wrapper_import,
)

from ..build import MODEL_REGISTRY
import src.utils.weight_init_helper as init_helper

from fairscale.nn.checkpoint import (
    checkpoint_wrapper,
)  

logger = logging.get_logger(__name__)

@MODEL_REGISTRY.register()
class SMViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg     
        t_work, h, w = cfg.MVIT.SPATIAL.PATCH_DIMS_WORK
    
        if cfg.MODEL.LONG_MEMORY_ENABLE:
            t_long, h, w = cfg.MVIT.SPATIAL.PATCH_DIMS_LONG
            t = t_long + t_work
            self.patch_dims = [t, h, w]
        else:
            self.patch_dims = [t_work, h, w] 

        # Prepare output.
        embed_dim = cfg.MVIT.EMBED_DIM  # base embed_dim
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS  # base head num
        mlp_ratio = cfg.MVIT.MLP_RATIO  # base ration
        qkv_bias = cfg.MVIT.QKV_BIAS  

        self.drop_rate = cfg.MVIT.DROPOUT_RATE  
        drop_path_rate = cfg.MVIT.DROPPATH_RATE  

        depth = cfg.MVIT.DEPTH  
        depth_spatial = cfg.MVIT.SPATIAL.DEPTH

        layer_scale_init_value = (
            cfg.MVIT.LAYER_SCALE_INIT_VALUE  # 0
        )  

        head_init_scale = cfg.MVIT.HEAD_INIT_SCALE  # 1.0
        mode = cfg.MVIT.MODE

        # Settings for feature Aggregation;
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL  

        if cfg.MVIT.NORM == "layernorm": 
            norm_layer = partial(nn.LayerNorm, eps=1e-6) 
        else:
            raise NotImplementedError("Only supports layernorm.")

        dpr = [
            x.item()
            for x in torch.linspace(
                0, drop_path_rate, depth
            ) 
        ][
            : cfg.MVIT.SPATIAL.DEPTH
        ]  # stochastic depth decay rule

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth_spatial + 1), torch.ones(
            depth_spatial + 1
        ) 

        for i in range(len(cfg.MVIT.SPATIAL.DIM_MUL)): 
            dim_mul[cfg.MVIT.SPATIAL.DIM_MUL[i][0]] = cfg.MVIT.SPATIAL.DIM_MUL[
                i
            ][1]
        for i in range(len(cfg.MVIT.SPATIAL.HEAD_MUL)):
            head_mul[
                cfg.MVIT.SPATIAL.HEAD_MUL[i][0]
            ] = cfg.MVIT.SPATIAL.HEAD_MUL[i][
                1
            ]  

        pool_q = [
            [] for i in range(cfg.MVIT.SPATIAL.DEPTH)
        ]  
        pool_kv = [[] for i in range(cfg.MVIT.SPATIAL.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.SPATIAL.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.SPATIAL.DEPTH)]

        for i in range(len(cfg.MVIT.SPATIAL.POOL_Q_STRIDE)):
            stride_q[
                cfg.MVIT.SPATIAL.POOL_Q_STRIDE[i][0]
            ] = cfg.MVIT.SPATIAL.POOL_Q_STRIDE[i][
                1:
            ]  

            if cfg.MVIT.SPATIAL.POOL_KVQ_KERNEL is not None:
                pool_q[
                    cfg.MVIT.SPATIAL.POOL_Q_STRIDE[i][0]
                ] = cfg.MVIT.SPATIAL.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.SPATIAL.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.SPATIAL.POOL_Q_STRIDE[i][1:]
                ] 
        # calculate the output patch dim of this spatial model;
        sample_rate_q = math.prod(torch.tensor(stride_q)[:, 0]).item()
        
        cfg.MVIT.TEMPORAL.PATCH_DIMS = [
            t_work,
            h // sample_rate_q,
            w // sample_rate_q,
        ]

        if cfg.MODEL.LONG_MEMORY_ENABLE:
            cfg.MVIT.COMPRESSOR.PATCH_DIMS = [
                t_long,
                h // sample_rate_q,
                w // sample_rate_q,
            ]
        # Adjust the KV Pooling Settings;
        if cfg.MVIT.SPATIAL.POOL_KV_STRIDE_ADAPTIVE is not None:
            
            _stride_kv = cfg.MVIT.SPATIAL.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.SPATIAL.POOL_KV_STRIDE = (
                []
            )  # Initialize the internal configurations;

            for i in range(cfg.MVIT.SPATIAL.DEPTH):
                if len(stride_q[i]) > 0: 
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.SPATIAL.POOL_KV_STRIDE.append([i] + _stride_kv)

        _stride_kv.insert(0, 1) 
        cfg.MVIT.TEMPORAL.POOL_KV_STRIDE_ADAPTIVE = (_stride_kv)  

        for i in range(len(cfg.MVIT.SPATIAL.POOL_KV_STRIDE)):
            stride_kv[
                cfg.MVIT.SPATIAL.POOL_KV_STRIDE[i][0]
            ] = cfg.MVIT.SPATIAL.POOL_KV_STRIDE[i][1:]
            if cfg.MVIT.SPATIAL.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.SPATIAL.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.SPATIAL.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.SPATIAL.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.SPATIAL.POOL_KV_STRIDE[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        input_size = self.patch_dims  # T H W

        self.blocks = nn.ModuleList()
        for i in range(depth_spatial):
            num_heads = round_width(num_heads, head_mul[i])  

            if (
                cfg.MVIT.DIM_MUL_IN_ATT
            ):  # True; introduce attention for Attention;
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1], 
                    divisor=round_width(
                        num_heads, head_mul[i + 1]
                    ), 
                )

            attention_block = MultiScaleBlock(
                dim=embed_dim, 
                dim_out=dim_out,
                num_heads=num_heads,  # 1
                input_size=input_size,  # [8, 56, 56]
                mlp_ratio=mlp_ratio,  # 4
                qkv_bias=qkv_bias,  # True
                drop_rate=self.drop_rate,  # 0.0
                drop_path=dpr[i], 
                norm_layer=norm_layer,  # layerNorm;
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i]
                if len(stride_kv) > i
                else [], 
                mode=mode,  
                pool_func=nn.Conv2d,
                has_cls_embed=self.cls_embed_on, 
                rel_pos_spatial=self.rel_pos_spatial, 
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT, 
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING, 
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT, 
                separate_qkv=cfg.MVIT.SEPARATE_QKV, 
            ) 

            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(
                    attention_block
                ) 

            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:

                input_size = [
                    size // stride
                    for size, stride in zip(input_size[1:], stride_q[i])
                ]  # Only Focus on the Spatial Size;
                input_size = [self.patch_dims[0]] + input_size
            embed_dim = dim_out 

        # Initialize for Temporal interation
        cfg.MVIT.TEMPORAL.NUM_HEADS = num_heads
        cfg.MVIT.TEMPORAL.EMBED_DIM = dim_out
        
        # Initialize for Temporal Compression
        cfg.MVIT.COMPRESSOR.NUM_HEADS = num_heads 
        cfg.MVIT.COMPRESSOR.EMBED_DIM = dim_out 

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.cls_embed_on:
                names.append("cls_token")
        return names

    def forward(self, work_inputs, bcthw):

        B, C, T, H, W = list(bcthw)  # [2, 96, 8, 56, 56] torch.Size([2, 96, 32, 56, 56])
        hw = [H, W] 

        for blk in self.blocks:
            work_inputs, hw = blk(work_inputs, hw)
        return work_inputs, hw 