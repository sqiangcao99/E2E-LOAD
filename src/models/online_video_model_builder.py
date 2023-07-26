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

import bisect
import numpy as np

import src.utils.logging as logging
import src.utils.weight_init_helper as init_helper
from src.models.batchnorm_helper import get_norm
from src.models.common import TwoStreamFusion
from src.models.utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
    validate_checkpoint_wrapper_import,
    # generate_square_subsequent_mask,
)

from . import head_helper, operators, resnet_helper, stem_helper  # noqa
from .build import MODEL_REGISTRY

from .modules import SMViT
from .modules import TMViTWork
from .modules import TMViTCompressor

from fairscale.nn.checkpoint import (
    checkpoint_wrapper,
) 

logger = logging.get_logger(__name__)


@MODEL_REGISTRY.register()
class STMViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Get parameters.

        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST  
        # Prepare input.
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]  # 3

        self.use_2d_patch = (
            cfg.MVIT.PATCH_2D
        )  

        self.patch_stride = cfg.MVIT.PATCH_STRIDE  
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.T_work = cfg.MODEL.WORK_MEMORY_NUM_SAMPLES  

        self.H = (
            cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        )   
        self.W = (
            cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        )  

        # Prepare output.
        self.num_classes = cfg.MODEL.NUM_CLASSES 

        embed_dim = cfg.MVIT.EMBED_DIM  
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS  
        mlp_ratio = cfg.MVIT.MLP_RATIO  
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH  # Time Depth And Space Depth;

        drop_path_rate = cfg.MVIT.DROPPATH_RATE   
        layer_scale_init_value = cfg.MVIT.LAYER_SCALE_INIT_VALUE  #
        head_init_scale = cfg.MVIT.HEAD_INIT_SCALE  #
        mode = cfg.MVIT.MODE  # conv
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL  
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL  

        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(
                nn.LayerNorm, eps=1e-6
            )  
        else:
            raise NotImplementedError("Only supports layernorm.")

        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,  # 3 7 7
            stride=cfg.MVIT.PATCH_STRIDE,  #
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch, 
        )  

        if cfg.MODEL.ACT_CHECKPOINT:  # Default False
            self.patch_embed = checkpoint_wrapper(self.patch_embed)

        self.patch_dims_work = [
            self.T_work,
            self.H,
            self.W,
        ]   

        cfg.MVIT.SPATIAL.PATCH_DIMS_WORK = self.patch_dims_work

        if cfg.MODEL.LONG_MEMORY_ENABLE:
            self.T_long = cfg.MODEL.LONG_MEMORY_NUM_SAMPLES  
            assert (self.T_work + self.T_long) == cfg.MODEL.TOTAL_MEMORY_NUM_SAMPLES
            
            self.patch_dims_long = [
                self.T_long,
                self.H,
                self.W,
            ]  

            cfg.MVIT.SPATIAL.PATCH_DIMS_LONG = self.patch_dims_long

        self.spatial_mvit = SMViT(cfg)  # dim_out = 384; for spatial and tempora; 
        if cfg.MODEL.LONG_MEMORY_ENABLE:
            self.temporal_compressor = TMViTCompressor(cfg)
        self.temporal_mvit = TMViTWork(cfg) 
        
        if self.cls_embed_on: 
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            ) 

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        logit_dim = cfg.MVIT.TEMPORAL.OUTPUT_DIM  
        
        self.norm = norm_layer(logit_dim) 
        self.norm_stem = (
            norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None
        )  # cfg.MVIT.NORM_STEM defaule false; 
        
        ## Define For Stream Inference; 
        self.memories_cache = None # 
        self.memories_compression_cache = None  
        self.temporal_steps = None  
        self.spatial_padding = None 

        self.head = head_helper.TransformerBasicHead(
            logit_dim,
            self.num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            cfg=cfg,
        )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
    
        self.apply(self._init_weights)

        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale) 

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
        self.memories_cache = None # 
        self.memories_compression_cache = None  
        self.temporal_steps = None  
        self.spatial_padding = None 
        self.temporal_mvit.empty_cache() 

        
    def forward(self, work_inputs, long_inputs=None, long_memory_mask=None):
        # (B C T H W) for 3D input, (B C H W) for 2D input;

        if self.cfg.MODEL.LONG_MEMORY_ENABLE: 
            
            work_inputs, bcthw_work = self.patch_embed(work_inputs, keep_spatial=True)  
            long_inputs, bcthw_long = self.patch_embed(long_inputs, keep_spatial=True)  
            
            assert self.T_work == work_inputs.shape[-3]
            assert self.T_long == long_inputs.shape[-3]

            work_inputs = einops.rearrange(work_inputs, "b c t h w -> (b t) (h w) c")  # torch.Size([64, 3136, 96])
            long_inputs = einops.rearrange(long_inputs, "b c t h w -> (b t) (h w) c")  # torch.Size([64, 3136, 96])
            
            B, C, T_work, H, W = bcthw_work
            B, C, T_long, H, W = bcthw_long

            if self.cls_embed_on:
                cls_tokens_work = self.cls_token.expand(
                    (B * (T_work)), -1, -1  
                )  # Expand for each frames; 
                cls_tokens_long = self.cls_token.expand(
                    (B * (T_long)), -1, -1  
                )   
                
                work_inputs = torch.cat((cls_tokens_work, work_inputs), dim=1)
                long_inputs = torch.cat((cls_tokens_long, long_inputs), dim=1)
            
            if self.cfg.MVIT.COMPRESSOR.DETACH:
                with torch.no_grad():
                    long_inputs, hw_long = self.spatial_mvit(long_inputs, bcthw_long) 
                
                long_inputs = long_inputs.detach() 
            else:
                raise NotImplementedError('To Do.')
                long_inputs, hw_long = self.spatial_mvit(long_inputs, bcthw_long)  
            
            work_inputs, hw_work = self.spatial_mvit(work_inputs, bcthw_work)  
            work_inputs = einops.rearrange(work_inputs, "(b t) hw c-> b t hw c", b=B, t=T_work)
            long_inputs = einops.rearrange(long_inputs, "(b t) hw c-> b t hw c", b=B, t=T_long)   

        else: 
            work_inputs, bcthw_work = self.patch_embed(work_inputs, keep_spatial=True)  
            B, C, T_work, H, W = list(bcthw_work) 
            work_inputs = einops.rearrange(work_inputs, "b c t h w -> (b t) (h w) c")  # torch.Size([64, 3136, 96])

            if self.cls_embed_on:
                cls_tokens = self.cls_token.expand(
                    (B * T_work), -1, -1  
                )  # Expand for each frames; 
                work_inputs = torch.cat((cls_tokens, work_inputs), dim=1)
            
            work_inputs, hw_work = self.spatial_mvit(work_inputs, bcthw_work) # bt hw c
            work_inputs = einops.rearrange(work_inputs, "(b t) hw c-> b t hw c", b=B, t=T_work) 

            # b t hw c -> b thw c
            work_inputs, hw_work = self.temporal_mvit(work_inputs)     
            work_inputs = einops.rearrange(work_inputs, "b (t hw) c-> b t hw c", t=hw_work[0])    
            
        # Process the long inputs; 
        if self.cfg.MODEL.LONG_MEMORY_ENABLE:
            if self.cls_embed_on and not self.cfg.MVIT.COMPRESSOR.CLS_TOKEN:
                long_inputs = long_inputs[:,:,1:] 

            long_inputs, hw_long = self.temporal_compressor(long_inputs) # Do Not Apply the shared backbone; 
            long_inputs = einops.rearrange(long_inputs, "b (t hw) c-> b t hw c", t=hw_long[0])
            # Perform Early Fusion
            
            if self.cfg.MODEL.LONG_MASK_ENABLE:
                long_compression_rate = int(torch.prod(torch.tensor(self.cfg.MVIT.COMPRESSOR.POOL_Q_STRIDE),dim=0)[1])
                long_memory_mask = einops.rearrange(long_memory_mask, 'b (t sr) -> b t sr', sr = long_compression_rate)
                long_memory_mask = torch.where(long_memory_mask>-1, 1, 0).sum(dim=-1)
                long_memory_mask = torch.where(long_memory_mask<1, -torch.inf, 0)
            else:
                long_memory_mask = None
            work_inputs, hw_work = self.temporal_mvit(work_inputs, long_inputs, long_memory_mask) # Do Not Apply the shared backbone; 
            work_inputs = einops.rearrange(work_inputs, "b (t hw) c-> b t hw c", t=hw_work[0])    

        if not (self.cfg.MVIT.SPATIAL.AGGREGATION.WORK_ENABLE):            
            if self.cfg.MVIT.SPATIAL.AGGREGATION.TYPE == "meanP":
                if self.cls_embed_on:
                    work_inputs = work_inputs[:, :, 1:, :]
                work_inputs = work_inputs.mean(-2)  
                
                work_inputs = self.norm(work_inputs)
            
            elif self.cfg.MVIT.SPATIAL.AGGREGATION.TYPE == "cls_token":
                work_inputs = self.norm(work_inputs)
                work_inputs = work_inputs[:, :, 0, :]
            else:
                work_inputs = self.norm(work_inputs)
                work_inputs = work_inputs.mean(-2)  
                          
        work_inputs = einops.rearrange(work_inputs, 'b t c-> (b t) c')
        scores = self.head(work_inputs) 
        scores = einops.rearrange(scores, '(b t) c-> b t c', t=self.T_work)
        return scores

    def stream_inference(self, work_inputs, long_indices, repeat_times, long_memory_mask=None):
        # (B C T H W) for 3D input, (B C H W) for 2D input;
 
        if not self.cfg.MODEL.LONG_MEMORY_ENABLE:
            self.T_long = 0
        else:
            self.temporal_steps = long_indices.clip(0)[-1]
        assert self.cls_embed_on
        # work_inputs.shape torch.Size([3, 96, 224, 224])
        ## B C T H W the patch
        if self.cfg.DATA.ZERO_MASK:
            if self.spatial_padding == None:
                spatial_padding = torch.zeros_like(work_inputs)    
                spatial_padding = spatial_padding[:,:,-self.cfg.MVIT.PATCH_KERNEL[0]:]  
            
                spatial_padding, bcthw = self.patch_embed(spatial_padding, keep_spatial=True)  
            
                B, C, T, H, W = bcthw # T = 1 
                spatial_padding = einops.rearrange(spatial_padding, "b c t h w -> (b t) (h w) c")  # torch.Size([64, 3136, 96])
                cls_tokens = self.cls_token.expand((B * (T)), -1, -1)      
                spatial_padding = torch.cat((cls_tokens, spatial_padding), dim=1)
                
                spatial_padding, _ = self.spatial_mvit(spatial_padding, bcthw)  
                self.spatial_padding = einops.rearrange(spatial_padding, "(b t) hw c-> b t hw c", b=B, t=T)
                
            # Downsampling the MASk
            if self.cfg.MODEL.LONG_MASK_ENABLE:
                long_compression_rate = int(torch.prod(torch.tensor(self.cfg.MVIT.COMPRESSOR.POOL_Q_STRIDE),dim=0)[1])
                long_memory_mask = einops.rearrange(long_memory_mask, 'b (t sr) -> b t sr', sr = long_compression_rate)
                long_memory_mask = torch.where(long_memory_mask>-1, 1, 0).sum(dim=-1)
                long_memory_mask = torch.where(long_memory_mask<1, -torch.inf, 0)

        work_inputs, bcthw = self.patch_embed(work_inputs, keep_spatial=True)  
        B, C, T, H, W = bcthw # T = 1 
        work_inputs = einops.rearrange(work_inputs, "b c t h w -> (b t) (h w) c")  # torch.Size([64, 3136, 96])
        cls_tokens = self.cls_token.expand((B * (T)), -1, -1)      
        work_inputs = torch.cat((cls_tokens, work_inputs), dim=1)
    
        work_inputs, _ = self.spatial_mvit(work_inputs, bcthw)  
        work_inputs = einops.rearrange(work_inputs, "(b t) hw c-> b t hw c", b=B, t=T)
        
        history_enable = False
         
        if self.memories_cache == None:
            self.memories_cache = work_inputs
            if self.cfg.MODEL.LONG_MEMORY_ENABLE:
                long_inputs = work_inputs[:,:1]
 
                long_indices = np.repeat(long_indices, repeat_times)
                sampled_long_indices = (long_indices % self.cfg.MODEL.LONG_MEMORY_SAMPLE_RATE == 0) 
                
                long_indices = long_indices[sampled_long_indices].clip(0) 
                long_indices = long_indices % (self.T_long * self.cfg.MODEL.LONG_MEMORY_SAMPLE_RATE) 
                long_inputs = long_inputs[:,long_indices] 
                history_enable = True

        elif self.memories_cache.shape[1] < (self.T_work + self.T_long * self.cfg.MODEL.LONG_MEMORY_SAMPLE_RATE):
 
            self.memories_cache = torch.cat((self.memories_cache, work_inputs), dim=1)
            work_inputs = self.memories_cache[:,-self.T_work:]
            
            if self.cfg.MODEL.LONG_MEMORY_ENABLE:
                long_inputs = self.memories_cache[:,:-self.T_work] 
                long_indices = np.repeat(long_indices, repeat_times) 
                sampled_long_indices = (long_indices % self.cfg.MODEL.LONG_MEMORY_SAMPLE_RATE == 0)
                long_indices = long_indices[sampled_long_indices].clip(0)
                long_indices = long_indices % (self.T_long * self.cfg.MODEL.LONG_MEMORY_SAMPLE_RATE) 
                long_inputs = long_inputs[:,long_indices]
                history_enable = ((self.temporal_steps % self.cfg.MODEL.LONG_MEMORY_SAMPLE_RATE) == 0 and self.temporal_steps != 0)
        else:
 
            assert self.memories_cache.shape[-3] == (self.T_work + self.T_long * self.cfg.MODEL.LONG_MEMORY_SAMPLE_RATE)
            self.memories_cache = torch.cat((self.memories_cache[:, 1:], work_inputs), dim=1) 
            work_inputs = self.memories_cache[:,-self.T_work:]
            
            if self.cfg.MODEL.LONG_MEMORY_ENABLE:
                long_inputs = self.memories_cache[:,:-self.T_work]
                long_indices = np.repeat(long_indices, repeat_times)
                # long_indices_ = long_indices
                assert long_inputs.shape[1] == len(long_indices)
                sampled_long_indices = (long_indices % self.cfg.MODEL.LONG_MEMORY_SAMPLE_RATE == 0) 
                long_inputs = long_inputs[:,sampled_long_indices]  
                # pdb.set_trace() 
                # long_inputs = long_inputs[:,long_indices]
                history_enable = ((self.temporal_steps % self.cfg.MODEL.LONG_MEMORY_SAMPLE_RATE) == 0)
 
        if self.cfg.DATA.ZERO_MASK:
            last_zero = bisect.bisect_right(long_indices, 0) - 1  
                          
            if last_zero > 0 :
                long_inputs[:,:last_zero] = self.spatial_padding 

        if self.cfg.MODEL.LONG_MEMORY_ENABLE:
            assert long_inputs.shape[1] == self.T_long
        
        assert work_inputs.shape[1] == self.T_work

        if self.cfg.MODEL.LONG_MEMORY_ENABLE:
            if self.cls_embed_on and not self.cfg.MVIT.COMPRESSOR.CLS_TOKEN:
                long_inputs = long_inputs[:,:,1:] 
                
            if history_enable:
                long_inputs, hw_long = self.temporal_compressor(long_inputs) # Do Not Apply the shared backbone; 
                long_inputs = einops.rearrange(long_inputs, "b (t hw) c-> b t hw c", t=hw_long[0])
                self.memories_compression_cache = long_inputs
            else:
                long_inputs = self.memories_compression_cache
            
            # Process the work inputs; 
            if self.cfg.DEMO.CACHE_INFERENCE: 
                work_inputs, hw_work = self.temporal_mvit.stream_inference(work_inputs, long_inputs) # Do Not Apply the shared backbone; 
            else:
                work_inputs, hw_work = self.temporal_mvit(work_inputs, long_inputs) # Do Not Apply the shared backbone; 
            work_inputs = einops.rearrange(work_inputs, "b (t hw) c-> b t hw c", t=hw_work[0])    
        else:
            if self.cfg.DEMO.CACHE_INFERENCE: 
                work_inputs, hw_work = self.temporal_mvit.stream_inference(work_inputs) # Do Not Apply the shared backbone; 
            else:
                work_inputs, hw_work = self.temporal_mvit(work_inputs) # Do Not Apply the shared backbone; 
    
            work_inputs = einops.rearrange(work_inputs, "b (t hw) c-> b t hw c", t=hw_work[0])    

        if not (self.cfg.MVIT.SPATIAL.AGGREGATION.WORK_ENABLE):  
            if self.cfg.MVIT.SPATIAL.AGGREGATION.TYPE == "meanP":
                if self.cls_embed_on:
                    work_inputs = work_inputs[:, :, 1:, :]
                work_inputs = work_inputs.mean(-2)  
                work_inputs = self.norm(work_inputs)
            elif self.cfg.MVIT.SPATIAL.AGGREGATION.TYPE == "cls_token":
                work_inputs = self.norm(work_inputs)
                work_inputs = work_inputs[:, :, 0, :]
            else:
                work_inputs = self.norm(work_inputs)
                work_inputs = work_inputs.mean(-2)  

        work_inputs = einops.rearrange(work_inputs, "b t c->(b t) c") 
        scores = self.head(work_inputs)
        scores = einops.rearrange(scores, "(b t) c->b t c", b=B, t=self.T_work)
        return scores
