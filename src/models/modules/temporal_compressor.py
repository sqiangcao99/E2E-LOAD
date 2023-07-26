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

import numpy as np 

import src.utils.logging as logging
import src.utils.weight_init_helper as init_helper

from src.models.batchnorm_helper import get_norm
from src.models.utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
    validate_checkpoint_wrapper_import,
)

from ..build import MODEL_REGISTRY
from .attention_compressor import MultiScaleBlock

from fairscale.nn.checkpoint import (
    checkpoint_wrapper,
) 
logger = logging.get_logger(__name__)

@MODEL_REGISTRY.register()
class TMViTCompressor(nn.Module): # Only Process The Current Models
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        
        # Get parameters.
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST  
        
        # Prepare input.
        if (cfg.MVIT.SPATIAL.AGGREGATION.LONG_ENABLE):  # (t h w) for spatial-temporal modeling;
            self.patch_dims = [
                cfg.MVIT.COMPRESSOR.PATCH_DIMS[0]
            ]  # Only retrive the temporal information;
            self.st = False  # only focuse on temporal modeling;
        else:  # t for temporal modeling only;
            self.patch_dims = cfg.MVIT.COMPRESSOR.PATCH_DIMS # [1, 14, 14]
            self.st = True
        
        num_patches = math.prod(
            self.patch_dims
        )  # cfg.MVIT.TEMPORAL.PATCH_DIMS [16, 14, 14]
        
        # Prepare backbone
        embed_dim = cfg.MVIT.COMPRESSOR.EMBED_DIM
        num_heads = cfg.MVIT.COMPRESSOR.NUM_HEADS

        mlp_ratio = cfg.MVIT.MLP_RATIO  # MLP
        qkv_bias = cfg.MVIT.QKV_BIAS  # learnable qkv bias items;
        self.drop_rate = cfg.MVIT.DROPOUT_RATE  # the drop rate of the networks

        depth = cfg.MVIT.DEPTH  # depth of the temporal model;
        depth_compressor = cfg.MVIT.COMPRESSOR.DEPTH # 
        drop_path_rate = cfg.MVIT.DROPPATH_RATE 
        layer_scale_init_value = (cfg.MVIT.LAYER_SCALE_INIT_VALUE)  # add adaptive for the res connection;
        head_init_scale = cfg.MVIT.HEAD_INIT_SCALE  #
        mode = cfg.MVIT.MODE
        
        self.cls_embed_on = cfg.MVIT.COMPRESSOR.CLS_TOKEN  
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(
                nn.LayerNorm, eps=1e-6
            )  
        else:
            raise NotImplementedError("Only supports layernorm.")

        dpr = [
            x.item()
            for x in torch.linspace(
                0, drop_path_rate, depth
            )  
        ][
            cfg.MVIT.COMPRESSOR.START_LAYER:
        ]  # stochastic depth decay rule; Fix this bugs; 

        dim_mul, head_mul = torch.ones(depth_compressor + 1), torch.ones(depth_compressor + 1)  # 初始化为1的列表;

        for i in range(len(cfg.MVIT.COMPRESSOR.DIM_MUL)):
            dim_mul[
                cfg.MVIT.COMPRESSOR.DIM_MUL[i][0]
            ] = cfg.MVIT.COMPRESSOR.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.COMPRESSOR.HEAD_MUL)):
            head_mul[
                cfg.MVIT.COMPRESSOR.HEAD_MUL[i][0]
            ] = cfg.MVIT.COMPRESSOR.HEAD_MUL[i][
                1
            ]  # expand the channel and dimension at specific layers; 

        pool_q = [[] for i in range(cfg.MVIT.COMPRESSOR.DEPTH)] # 
        pool_kv = [[] for i in range(cfg.MVIT.COMPRESSOR.DEPTH)] #  
        stride_q = [[] for i in range(cfg.MVIT.COMPRESSOR.DEPTH)] #  
        stride_kv = [[] for i in range(cfg.MVIT.COMPRESSOR.DEPTH)] 
        
        # Initialize the query kernels; 
        for i in range(len(cfg.MVIT.COMPRESSOR.POOL_Q_STRIDE)):
            
            stride_q[cfg.MVIT.COMPRESSOR.POOL_Q_STRIDE[i][0]] = cfg.MVIT.COMPRESSOR.POOL_Q_STRIDE[i][1:] 
            if cfg.MVIT.COMPRESSOR.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.COMPRESSOR.POOL_Q_STRIDE[i][0]] = cfg.MVIT.COMPRESSOR.POOL_KVQ_KERNEL 
            else:
                pool_q[cfg.MVIT.COMPRESSOR.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.TEMPORAL.POOL_Q_STRIDE[i][1:]]

        sample_rate_q = torch.prod(torch.tensor(stride_q), dim = 0).tolist()
        cfg.MVIT.FUSION.KEYVALUE_PATCH_DIMS = [
            self.patch_dims[0] // int(sample_rate_q[0]),
            self.patch_dims[1] // int(sample_rate_q[1]),
            self.patch_dims[2] // int(sample_rate_q[2]),
        ]  
    
        if len(cfg.MVIT.COMPRESSOR.POOL_KV_STRIDE_ADAPTIVE) == 0:  
            cfg.MVIT.COMPRESSOR.POOL_KV_STRIDE_ADAPTIVE = np.prod(stride_q,axis=0).tolist()
        
        # Initialize according the Query Downsamping Path; 
        if cfg.MVIT.COMPRESSOR.POOL_KV_STRIDE_ADAPTIVE is not None:
            
            _stride_kv = (cfg.MVIT.COMPRESSOR.POOL_KV_STRIDE_ADAPTIVE)  
            cfg.MVIT.COMPRESSOR.POOL_KV_STRIDE = ([])  # Initialize the internal configurations;

            for i in range(cfg.MVIT.COMPRESSOR.DEPTH): 
                if len(stride_q[i]) > 0:  
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.COMPRESSOR.POOL_KV_STRIDE.append([i] + _stride_kv) 
            
        for i in range(len(cfg.MVIT.COMPRESSOR.POOL_KV_STRIDE)):

            stride_kv[
                cfg.MVIT.COMPRESSOR.POOL_KV_STRIDE[i][0]
            ] = cfg.MVIT.COMPRESSOR.POOL_KV_STRIDE[i][1:]
            if cfg.MVIT.COMPRESSOR.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.COMPRESSOR.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.COMPRESSOR.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.COMPRESSOR.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.COMPRESSOR.POOL_KV_STRIDE[i][1:]
                ]

        self.pool_q = pool_q 
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv
        
        input_size = self.patch_dims  
        self.blocks = nn.ModuleList()
        
        for i in range(depth_compressor):
            num_heads = round_width(num_heads, head_mul[i])  
            
            if cfg.MVIT.DIM_MUL_IN_ATT:  # True;
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
                cfg,
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
                pool_func=nn.Conv1d
                if cfg.MVIT.SPATIAL.AGGREGATION.WORK_ENABLE
                else nn.Conv3d,
                has_cls_embed=self.cls_embed_on,  
                pool_first=pool_first,   
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT, 
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING,  
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,  
                separate_qkv=False,)
            
            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(
                    attention_block
                )

            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride
                    for size, stride in zip(input_size, stride_q[i])
                ]
            
            embed_dim = dim_out   
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
            names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")
        return names

    def forward(self, x, mask=None):
     
        B, T, HW, C = x.size()  
        
        thw = self.cfg.MVIT.COMPRESSOR.PATCH_DIMS # 8 14 14 
        x = einops.rearrange(
            x, "b t hw c-> b (t hw) c"
        ) 
        for blk in self.blocks:
            x, thw = blk(x, thw)
        return x, thw