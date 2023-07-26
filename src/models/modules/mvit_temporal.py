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
from src.models.common import TwoStreamFusion
from src.models.reversible_mvit import ReversibleMViT
from src.models.utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
    validate_checkpoint_wrapper_import,
)

from ..build import MODEL_REGISTRY
from .attention_temporal import MultiScaleBlock

from fairscale.nn.checkpoint import (
    checkpoint_wrapper,
)  

logger = logging.get_logger(__name__)


@MODEL_REGISTRY.register()
class TMViTWork(nn.Module): # Only Process The Current Models
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        # Get parameters.
        self.cfg = cfg
        # Prepare input.
        if (
            cfg.MVIT.SPATIAL.AGGREGATION.WORK_ENABLE
        ):  # (t h w) for spatial-temporal modeling;
            self.patch_dims = [
                cfg.MVIT.TEMPORAL.PATCH_DIMS[0]
            ]  # Only retrive the temporal information;
            self.st = False  # only focuse on temporal modeling;
        else:  # t for temporal modeling only;
            self.patch_dims = cfg.MVIT.TEMPORAL.PATCH_DIMS
            self.st = True
        
        # Prepare backbone
        embed_dim = cfg.MVIT.TEMPORAL.EMBED_DIM
        num_heads = cfg.MVIT.TEMPORAL.NUM_HEADS

        mlp_ratio = cfg.MVIT.MLP_RATIO  # MLP
        qkv_bias = cfg.MVIT.QKV_BIAS  # learnable qkv bias items;
        self.drop_rate = cfg.MVIT.DROPOUT_RATE  # the drop rate of the networks

        depth = cfg.MVIT.DEPTH 
        depth_temporal = cfg.MVIT.TEMPORAL.DEPTH 
        
        drop_path_rate = cfg.MVIT.DROPPATH_RATE 
        layer_scale_init_value = cfg.MVIT.LAYER_SCALE_INIT_VALUE  # scale the resnet connection; 
        head_init_scale = cfg.MVIT.HEAD_INIT_SCALE  

        mode = cfg.MVIT.MODE 
        
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON # True; 
        self.rel_pos_spatial = (cfg.MVIT.TEMPORAL.REL_POS_SPATIAL)  
        self.rel_pos_temporal = (cfg.MVIT.TEMPORAL.REL_POS_TEMPORAL)  
    
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(
                nn.LayerNorm, eps=1e-6
            )  
        else:
            raise NotImplementedError("Only supports layernorm.")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)][cfg.MVIT.SPATIAL.DEPTH:]  # 

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth_temporal + 1), torch.ones(
            depth_temporal + 1
        )  

        for i in range(len(cfg.MVIT.TEMPORAL.DIM_MUL)):
            dim_mul[
                cfg.MVIT.TEMPORAL.DIM_MUL[i][0]
            ] = cfg.MVIT.TEMPORAL.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.TEMPORAL.HEAD_MUL)):
            head_mul[
                cfg.MVIT.TEMPORAL.HEAD_MUL[i][0]
            ] = cfg.MVIT.TEMPORAL.HEAD_MUL[i][
                1
            ] 
        
        
        pool_q = [[] for i in range(cfg.MVIT.TEMPORAL.DEPTH)] 
        pool_kv = [[] for i in range(cfg.MVIT.TEMPORAL.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.TEMPORAL.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.TEMPORAL.DEPTH)]

        for i in range(len(cfg.MVIT.TEMPORAL.POOL_Q_STRIDE)):
            stride_q[
                cfg.MVIT.TEMPORAL.POOL_Q_STRIDE[i][0]
            ] = cfg.MVIT.TEMPORAL.POOL_Q_STRIDE[i][
                1:
            ]  
            if cfg.MVIT.TEMPORAL.POOL_KVQ_KERNEL is not None:
                pool_q[
                    cfg.MVIT.TEMPORAL.POOL_Q_STRIDE[i][0]
                ] = cfg.MVIT.TEMPORAL.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.TEMPORAL.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.TEMPORAL.POOL_Q_STRIDE[i][1:]
                ] 

        sample_rate_q = torch.prod(torch.tensor(stride_q), dim = 0) 
        cfg.MVIT.FUSION.QUERY_PATCH_DIMS = [
            self.patch_dims[0] // int(sample_rate_q[0]),
            self.patch_dims[1] // int(sample_rate_q[1]),
            self.patch_dims[2] // int(sample_rate_q[2]),
        ] 

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.TEMPORAL.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = (
                cfg.MVIT.TEMPORAL.POOL_KV_STRIDE_ADAPTIVE
            )  
            cfg.MVIT.TEMPORAL.POOL_KV_STRIDE = (
                []
            )  # Initialize the internal configurations;
            for i in range(cfg.MVIT.TEMPORAL.DEPTH):
                if len(stride_q[i]) > 0:  
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.TEMPORAL.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.TEMPORAL.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.TEMPORAL.POOL_KV_STRIDE[i][0]] = cfg.MVIT.TEMPORAL.POOL_KV_STRIDE[i][1:]
            if cfg.MVIT.TEMPORAL.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.TEMPORAL.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.TEMPORAL.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.TEMPORAL.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.TEMPORAL.POOL_KV_STRIDE[i][1:]
                ]

        self.pool_q = pool_q 
        self.pool_kv = pool_kv 
        self.stride_q = stride_q 
        self.stride_kv = stride_kv 

        input_size = self.patch_dims 
        self.blocks = nn.ModuleList() 

        if cfg.MODEL.LONG_MEMORY_ENABLE: #
            fusion_layers = np.zeros(depth_temporal) # 
            fusion_layers[cfg.MVIT.FUSION.EARLY_FUSION_LAYERS] = 1
            self.fusion_layers = fusion_layers==1 #
        
        else:
            self.fusion_layers = np.zeros(depth_temporal) == 1

        for i in range(depth_temporal):
            num_heads = round_width(num_heads, head_mul[i])  #
            
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
                rel_pos_spatial=self.rel_pos_spatial,  
                rel_pos_temporal=self.rel_pos_temporal, 
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT, 
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING, 
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,  
                separate_qkv=cfg.MVIT.SEPARATE_QKV, 
                early_fusion = self.fusion_layers[i] 
            ) 

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

        cfg.MVIT.TEMPORAL.OUTPUT_DIM = dim_out  
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
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def empty_cache(self):
        for blk in self.blocks:
            blk.empty_cache() 

    def forward(self, x, long_inputs=None, long_memory_mask=None):
        # B, T, HW, C = x.size()
        thw = self.cfg.MVIT.TEMPORAL.PATCH_DIMS 
        x = einops.rearrange(
            x, "b t hw c-> b (t hw) c"
        )  # with cls token; need to be removed when pooling;

        for blk in self.blocks: # 
            x, thw = blk(x, long_inputs, thw,long_memory_mask=long_memory_mask)  
        return x, thw

    def stream_inference(self, x, long_inputs=None, long_memory_mask=None):

        if self.cfg.MVIT.SPATIAL.AGGREGATION.WORK_ENABLE:
            B, T, C = x.size()
            thw = [T]
        else:

            B, T, HW, C = x.size()
            # thw = [T, H, W]
            thw = self.cfg.MVIT.TEMPORAL.PATCH_DIMS
            x = einops.rearrange(
                x, "b t hw c-> b (t hw) c"
            )  # with cls token; need to be removed when pooling;
        for blk in self.blocks:
            x, thw = blk.stream_inference(x, long_inputs, thw)
        return x, thw