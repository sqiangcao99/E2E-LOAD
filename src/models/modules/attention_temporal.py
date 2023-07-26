#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import pdb
import einops

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from src.models.common import DropPath, Mlp
from src.models.utils import generate_casual_mask

from .basic_attention import AttentionBlock

def attention_pool(
    tensor,
    pool,
    thw_shape,
    has_cls_embed=True,
    norm=None,
    spatial_temporal=None,
):
    """
    tensor: ([2, 1, 25089, 96]);
    pool: Conv3d(96, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=96, bias=False)
    thw_shape: [8, 56, 56]
    has_cls_embed: True
    norm: LayerNorm((96,), eps=1e-06, elementwise_affine=True)
    """

    if pool is None:  
        return tensor, thw_shape

    tensor_dim = tensor.ndim  # 4;
    if tensor_dim == 4:  
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1) 
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed and spatial_temporal:
        tensor = einops.rearrange(
            tensor, "b n (t hw) c -> b n t hw c", t=thw_shape[0]
        ) 
        cls_tok, tensor = (
            tensor[:, :, :, :1, :],
            tensor[:, :, :, 1:, :],
        )  # tensor.shape: 

        B, N, T, HW, C = tensor.shape 
        T, H, W = thw_shape #
        tensor = einops.rearrange(tensor, "b nh t (h w) c -> (b nh) c t h w", h=H, w=W)
    else:
        raise NotImplementedError('Not implemented')

    tensor = pool(tensor)
    
    # torch.Size([2, 96, 8, 56, 56])
    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]

    if has_cls_embed and spatial_temporal:
        tensor = einops.rearrange(tensor, "(b n) c t h w -> b n t (h w) c", n=N)
        tensor = torch.cat(
            (cls_tok, tensor), dim=-2
        )  # torch.Size([1, 4, 16, 197, 96])
        tensor = einops.rearrange(tensor, "b n t hw c -> b n (t hw) c")
    else:
        pdb.set_trace() 

    if norm is not None:
        tensor = norm(tensor) 
    
    if tensor_dim == 4:
        pass
    else:  
        tensor = tensor.squeeze(1)

    return tensor, thw_shape  # ([1, 4, 3152, 96]), [16, 14, 14]


def get_rel_pos(rel_pos, d):
    if isinstance(d, int): 
        ori_d = rel_pos.shape[0] 
        if ori_d == d:
            return rel_pos 
        else:
            # Interpolate rel pos.
            new_pos_embed = F.interpolate( 
                rel_pos.reshape(1, ori_d, -1).permute(0, 2, 1),
                size=d,
                mode="linear",
            )

            return new_pos_embed.reshape(-1, d).permute(1, 0)


def cal_rel_pos_spatial(
    attn, q, k, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w
):
    """
    Decomposed Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0 
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape  #

    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1) 

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)

    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio 
        - torch.arange(k_h)[None, :] * k_h_ratio 
    ) 

    dist_h += (k_h - 1) * k_h_ratio 

    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0) 
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio
        - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    # Intepolate rel pos if needed.
    rel_pos_h = get_rel_pos(rel_pos_h, dh)
    rel_pos_w = get_rel_pos(rel_pos_w, dw) 
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]  # Apply indices selection;

    B, n_head, q_N, dim = q.shape  # torch.Size([2, 1, 25089, 96])

    r_q = einops.rearrange(q, "b n (t hw) c-> b n t hw c", t=q_shape[0])
    cls_tok, r_q = (
        r_q[:, :, :, :1, :],
        r_q[:, :, :, 1:, :],
    ) 
    r_q = einops.rearrange(r_q, "b n t (h w) c-> b n t h w c", h=q_shape[1])

    rel_h_q = torch.einsum(
        "bythwc,hkc->bythwk", r_q, Rh
    ) 
 
    rel_w_q = torch.einsum(
        "bythwc,wkc->bythwk", r_q, Rw
    )  

    rel_h_q = rel_h_q[:, :, :, :, :, None, :, None].expand(
        -1, -1, -1, -1, -1, k_shape[0], -1, k_shape[2]
    )
    rel_w_q = rel_w_q[:, :, :, :, :, None, None, :].expand(
        -1, -1, -1, -1, -1, k_shape[0], k_shape[1], -1
    )

    rel_h_q = einops.rearrange(
        rel_h_q, "b n q_t q_h q_w k_t k_h k_w-> b n q_t (q_h q_w) k_t (k_h k_w)"
    )
    rel_w_q = einops.rearrange(
        rel_w_q, "b n q_t q_h q_w k_t k_h k_w-> b n q_t (q_h q_w) k_t (k_h k_w)"
    )

    attn = einops.rearrange(
        attn,
        "b n (q_t q_hw) (k_t k_hw) -> b n q_t q_hw k_t k_hw",
        q_t=q_shape[0],
        k_t=k_shape[0],
    )
    # pdb.set_trace() 

    if has_cls_embed:
        attn[:, :, :, 1:, :, 1:] = attn[:, :, :, 1:, :, 1:] + rel_h_q + rel_w_q
    else:
        attn[:, :, :, 1:, :, :] = attn[:, :, :, 1:, :, :] + rel_h_q + rel_w_q
    
    attn = einops.rearrange(
        attn, "b n q_t q_hw k_t k_hw -> b n (q_t q_hw) (k_t k_hw)"
    )

    return attn

def cal_rel_pos_temporal(
    attn, q, has_cls_embed, q_shape, k_shape, rel_pos_t
):
    """
    Temporal Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0

    if len(q_shape) == 1:
        assert len(q_shape) == len(k_shape)
        q_t = q_shape[0]
        k_t = k_shape[0]
        q_h = 1
        q_w = 1
        k_h = 1
        k_w = 1
    else:
        q_t, q_h, q_w = q_shape
        k_t, k_h, k_w = k_shape

    dt = int(2 * max(q_t, k_t) - 1)  #
    # Intepolate rel pos if needed.
    rel_pos_t = get_rel_pos(rel_pos_t, dt)

    q_t_ratio = 1
    k_t_ratio = 1

    dist_t = (
        torch.arange(k_t)[:, None] * k_t_ratio
        - torch.arange(k_t)[None, :] * k_t_ratio
    )
    
    dist_t += (k_t - 1) * k_t_ratio 
    Rt = rel_pos_t[dist_t.long()] 
    B, n_head, q_N, dim = q.shape  # torch.Size([1, 4, 3152, 96])
    r_q = einops.rearrange(q, "b n (t hw) c-> t (b n hw) c", t=q_shape[0])
    
    if q_t == 1: 
        Rt = Rt[-1:]
        rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(
            0, 1
        ) 
     
    else:
        rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(
            0, 1
        )  # Rt.transpose(1, 2) # 788 16 16
       
    rel = einops.repeat(
        rel,
        "(b n q_hw) q_t k_t -> b n (q_t q_hw) (k_t k_hw)",
        b=B,
        n=n_head,
        q_hw=q_shape[1] * q_shape[2] + 1,
        k_hw=k_shape[1] * k_shape[2] + 1,
    )

    attn = attn + rel
    return attn


class MultiScaleAttention(nn.Module): 
    def __init__(
        self,
        cfg,
        dim,
        dim_out,
        input_size,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",  # 3D or 2D conv?
        pool_func=nn.Conv3d,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        rel_pos_zero_init=False,
        residual_pooling=False,
        separate_qkv=False,
        early_fusion=False,
        
    ):
        super().__init__()
        self.separate_qkv = separate_qkv  # False
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads  
        self.scale = head_dim**-0.5  
        self.has_cls_embed = has_cls_embed  
        self.mode = mode  # CONV Pooling Mode;
        self.early_fusion = early_fusion 

        padding_q = [int(q // 2) for q in kernel_q]  
        padding_kv = [int(kv // 2) for kv in kernel_kv] 

        padding_q = [int(q // 2) for q in kernel_q] 
        padding_kv = [int(kv // 2) for kv in kernel_kv] 
        assert padding_q[0] == 0
        assert padding_kv[0] == 0 

        assert kernel_kv[0] == 1
        assert kernel_q[0] == 1
            
        
        self.st = not (cfg.MVIT.SPATIAL.AGGREGATION.WORK_ENABLE) 
        self.cfg = cfg

        if separate_qkv: 
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)  #

        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        # Fix the convoluational kernels here;
        # disable the conv os at the certain situations;å
        if mode in ("avg", "max"):
            if cfg.MVIT.SPATIAL.AGGREGATION.ENABLE:
                pool_op = nn.MaxPool1d if mode == "max" else nn.AvgPool1d
            else:
                pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d

            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv" or mode == "conv_unshared":

            dim_conv = dim_out // num_heads if mode == "conv" else dim_out

            self.pool_q = (
                pool_func(
                    dim_conv,  #
                    dim_conv, 
                    kernel_q, 
                    stride=stride_q,  # stride on three dimension;
                    padding=padding_q,
                    groups=dim_conv,  #
                    bias=False,
                )
                if len(kernel_q)
                > 0 
                else None
            )  # Conv3d(96, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=96, bias=False)

            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None

            self.pool_k = (
                pool_func(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None

            self.pool_v = (
                pool_func(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            ) 
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        self.rel_pos_spatial = (
            rel_pos_spatial and not cfg.MVIT.SPATIAL.AGGREGATION.WORK_ENABLE
        )
        self.rel_pos_temporal = rel_pos_temporal

        if self.rel_pos_spatial: 

            assert input_size[1] == input_size[2]  # [32, 56]
            size = input_size[1]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = (
                size // stride_kv[1] if len(stride_kv) > 0 else size
            ) 
            rel_sp_dim = 2 * max(q_size, kv_size) - 1 

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(rel_sp_dim, head_dim)
            )
            if not rel_pos_zero_init: 
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

        if self.rel_pos_temporal:
            self.rel_pos_t = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim)
            )  #
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_t, std=0.02)

        self.residual_pooling = residual_pooling
        # the cached memories 
        self.q_cache = torch.empty(0)
        self.k_cache = torch.empty(0)
        self.v_cache = torch.empty(0)
        self.x_cache = torch.empty(0)
    
    def forward(self, x, thw_shape, long_inputs=None, long_memory_mask=None):

        B, N, _ = x.shape  # fix the shape here;

        assert self.mode != "conv_unshared"
        if not self.separate_qkv: 
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, -1)
                .permute(2, 0, 3, 1, 4) 
            )
            q, k, v = qkv[0], qkv[1], qkv[2] 
            # q.shape: ([2, 4, 32, 96])
            if self.cfg.MODEL.LONG_MEMORY_ENABLE:
                if self.early_fusion:
                    # pdb.set_trace() 
                    if self.cfg.MVIT.FUSION.EARLY_FUSION_TYPE=='kv': 
                        qkv_long = self.qkv(long_inputs) 
                        # pdb.set_trace()
                        qkv_long = einops.rearrange(qkv_long, 'b thw (na nh c) -> na b nh thw c', na=3, nh=self.num_heads) 
                        q_long, k_long, v_long = qkv_long[0], qkv_long[1], qkv_long[2]  # the projected Q、K、V;

        else: 
            q = k = v = x
            q = (
                self.q(q)
                .reshape(B, N, self.num_heads, -1)
                .permute(0, 2, 1, 3) 
            )

            k = (
                self.k(k)
                .reshape(B, N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )
            v = (
                self.v(v)
                .reshape(B, N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )

        q, q_shape = attention_pool(
            q,
            self.pool_q,  # Conv3d(96, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=96, bias=False)
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
            spatial_temporal=self.st,
        )

        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
            spatial_temporal=self.st,
        )  # k 393:
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v
            if hasattr(self, "norm_v")
            else None,  
            spatial_temporal=self.st,
        )

        N = q.shape[2] 

        if self.cfg.MODEL.LONG_MEMORY_ENABLE:
            # pdb.set_trace()
            if self.early_fusion:
                if self.cfg.MVIT.FUSION.EARLY_FUSION_TYPE=='kv':
                    
                    attn_long = (q * self.scale) @ k_long.transpose(
                        -2, -1
                    ) 
                    # pdb.set_trace()

                    if self.cfg.MODEL.LONG_MASK_ENABLE:
                        
                        long_memory_mask = einops.repeat(
                            long_memory_mask,
                            'b k_t -> b h (q_t q_hw) (k_t k_hw)' ,
                            q_t=32, 
                            q_hw=14 * 14 + 1,
                            k_hw=7 * 7,
                            h=1,
                        ) 
                        
                        attn_long = attn_long + long_memory_mask
                        

        attn = (q * self.scale) @ k.transpose(
            -2, -1
        )  # torch.Size([2, 1, 25089, 393]) # raw input without function activate;

        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                k,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h,  # [27, 96]
                self.rel_pos_w,  
            )
            if self.cfg.MODEL.LONG_MEMORY_ENABLE:
                if self.early_fusion:
                    
                    if self.cfg.MVIT.FUSION.EARLY_FUSION_TYPE=='kv':
                        k_long_shape = [8,7,7] # 
                        attn_long = cal_rel_pos_spatial(
                            attn_long,
                            q,
                            k_long,
                            False, 
                            q_shape,
                            k_long_shape,
                            self.rel_pos_h,
                            self.rel_pos_w,
                        )

        if self.rel_pos_temporal:
            attn = cal_rel_pos_temporal(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_t, 
            )
        
        if self.cfg.MODEL.CASUAL_MASK_ENABLE:
            assert q_shape[0] == k_shape[0]
            casual_mask = generate_casual_mask(q_shape[0], device=attn.device)

            # 16 * (14 14) * 16 * 7 * 7q
            casual_mask = einops.repeat(
                casual_mask,
                "q_t k_t -> b h (q_t q_hw) (k_t k_hw)",
                q_hw=q_shape[1] * q_shape[2] + 1,
                k_hw=k_shape[1] * k_shape[2] + 1,
                b=1,
                h=1,
            ) 
            attn = attn + casual_mask 

        if self.early_fusion:                
            if self.cfg.MVIT.FUSION.EARLY_FUSION_TYPE=='kv':
                attn = torch.cat((attn, attn_long), dim=-1) 
                v = torch.cat((v, v_long), dim=-2)

        attn = attn.softmax(dim=-1)
        x = attn @ v

        if self.residual_pooling: 
            x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x, q_shape

    def stream_inference(self, x, thw_shape, long_inputs=None):
        B, N, _ = x.shape  # fix the shape here;
        assert self.mode != "conv_unshared"
        if not self.separate_qkv:  
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, -1)
                .permute(2, 0, 3, 1, 4)   
            )
            q, k, v = qkv[0], qkv[1], qkv[2]  # the projected Q、K、V;
            # q.shape: ([2, 4, 32, 96])
            
            if self.early_fusion:
                if self.cfg.MVIT.FUSION.EARLY_FUSION_TYPE=='kv':
                    qkv_long = self.qkv(long_inputs) 

                    qkv_long = einops.rearrange(qkv_long, 'b thw (na nh c) -> na b nh thw c', na=3, nh=self.num_heads)
                    q_long, k_long, v_long = qkv_long[0], qkv_long[1], qkv_long[2]  # the projected Q、K、V;

        else:  
            q = k = v = x
            q = (
                self.q(q)
                .reshape(B, N, self.num_heads, -1)
                .permute(0, 2, 1, 3)  
            )

            k = (
                self.k(k)
                .reshape(B, N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )
            v = (
                self.v(v)
                .reshape(B, N, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )

        # pdb.set_trace()
        q, q_shape = attention_pool(
            q,
            self.pool_q,  # Conv3d(96, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=96, bias=False)
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
            spatial_temporal=self.st,
        )

        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
            spatial_temporal=self.st,
        )  # k 393:
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v
            if hasattr(self, "norm_v")
            else None,   
            spatial_temporal=self.st,
        )

       
        N = q.shape[2]   

        if not len(self.x_cache) == 0:
            q = einops.rearrange(q, 'b nh (t hw) c -> b nh t hw c', t=q_shape[0])
            q = q[:,:,-1]  
            q_shape[0] = 1
            # pdb.set_trace() 

        if self.early_fusion:
            if self.cfg.MVIT.FUSION.EARLY_FUSION_TYPE=='kv':
                
                attn_long = (q * self.scale) @ k_long.transpose(
                    -2, -1) # ([1, 4, 6304, 392])

        attn = (q * self.scale) @ k.transpose(
            -2, -1
        )  # torch.Size([2, 1, 25089, 393]) # raw input without function activate;
        
        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                k,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h,  # [27, 96]
                self.rel_pos_w,   
            ) 

            if self.early_fusion:
                # pdb.set_trace() 
                if self.cfg.MVIT.FUSION.EARLY_FUSION_TYPE=='kv':
                    k_long_shape = [8,7,7]
                    attn_long = cal_rel_pos_spatial(
                        attn_long,
                        q,
                        k_long,
                        False, 
                        q_shape,
                        k_long_shape,
                        self.rel_pos_h,  # [27, 96]
                        self.rel_pos_w,  
                    )  

        if self.rel_pos_temporal:
            attn = cal_rel_pos_temporal(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_t,  
            )

        if len(self.x_cache) == 0: 
            if self.cfg.MODEL.CASUAL_MASK_ENABLE:

                assert q_shape[0] == k_shape[0]
                casual_mask = generate_casual_mask(q_shape[0], device=attn.device)

                # 16 * (14 14) * 16 * 7 * 7q
                casual_mask = einops.repeat(
                    casual_mask,
                    "q_t k_t -> b h (q_t q_hw) (k_t k_hw)",
                    q_hw=q_shape[1] * q_shape[2] + 1,
                    k_hw=k_shape[1] * k_shape[2] + 1,
                    b=1,
                    h=1,
                ) 
                attn = attn + casual_mask 

        if self.early_fusion:
            if self.cfg.MVIT.FUSION.EARLY_FUSION_TYPE=='kv': # False

                attn = torch.cat((attn, attn_long), dim=-1)
                v = torch.cat((v, v_long), dim=-2)

        attn = attn.softmax(dim=-1)
        x = attn @ v 

        if self.residual_pooling:  
            x = x + q
        else:
            raise NotImplementedError

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        if len(self.x_cache) == 0:
            x = einops.rearrange(x, 'b (t hw) c-> b t hw c', t = q_shape[0])
            self.x_cache = x
            x = einops.rearrange(x, 'b t hw c-> b (t hw) c')
        else:
            x = einops.rearrange(x, 'b (t hw) c-> b t hw c', t = 1)
            self.x_cache = torch.cat((self.x_cache[:,1:], x), dim=1)  
            x = einops.rearrange(self.x_cache, 'b t hw c-> b (t hw) c')  

            # repair the shape of q
            q_shape[0] = k_shape[0]

        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape


class MultiScaleBlock(nn.Module):  
    def __init__(
        self,
        cfg,
        dim,
        dim_out,
        num_heads,
        input_size,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        layer_scale_init_value=0.0, 
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        pool_func=nn.Conv3d,
        has_cls_embed=True,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        rel_pos_zero_init=False,
        residual_pooling=False,
        dim_mul_in_att=False,
        separate_qkv=False,

        early_fusion=False
    ):
        super().__init__()
        
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim) 
        self.dim_mul_in_att = dim_mul_in_att  # True
        self.early_fusion = early_fusion  

        self.st = not (cfg.MVIT.SPATIAL.AGGREGATION.WORK_ENABLE)
        self.cfg = cfg

        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]  # ? [1, 1, 1]
        stride_skip = stride_q  # [1, 1, 1]
        padding_skip = [int(skip // 2) for skip in kernel_skip]  # ? [0, 0, 0]

        att_dim = (
            dim_out if dim_mul_in_att else dim
        )  

        # calculate the kernel and stride;
        if (
            numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1
        ):   
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()  

        if cfg.MVIT.SPATIAL.AGGREGATION.WORK_ENABLE:
            kernel_q = ()
            kernel_kv = ()
        else:

            if len(kernel_q) != 0:
                kernel_q[0] = 1 
                assert stride_q[0] == 1 
            else:
                pdb.set_trace()

            if len(kernel_kv) != 0: 
                kernel_kv[0] = 1
                assert stride_kv[0] == 1
            else:
                pdb.set_trace()

            # enabling 3d Conv3d, for conenients
        self.attn = MultiScaleAttention(
            cfg,
            dim, 
            att_dim, 
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_func=pool_func,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_temporal=rel_pos_temporal,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
            separate_qkv=separate_qkv,
            early_fusion=early_fusion,
        )

        if self.early_fusion and cfg.MODEL.LONG_MEMORY_ENABLE:
            
            if cfg.MVIT.FUSION.EARLY_FUSION_TYPE == 'ca':
                patch_dims_q = input_size
                patch_dims_kv = cfg.MVIT.FUSION.KEYVALUE_PATCH_DIMS
                
                self.fusion_block = AttentionBlock(
                    cfg,
                    dim=dim, 
                    kv_dim = cfg.MVIT.FUSION.INTERACTION.EMBED_DIM[0], 
                    dim_out=att_dim,
                    num_heads=num_heads, 
                    q_size=patch_dims_q,
                    kv_size=patch_dims_kv,  
                    mlp_ratio=mlp_ratio,  # 4
                    qkv_bias=qkv_bias,  # True
                    drop_rate=drop_path,  # 
                    layer_scale_init_value=cfg.MVIT.FUSION.INTERACTION.LAYER_SCALE_INIT_VALUE,
                    norm_layer=norm_layer,  # layerNorm;
                    has_cls_embed=has_cls_embed, 
                    rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT, 
                    dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                    separate_qkv=True, 
                    operator='cross_attention', 
                ) 
            elif cfg.MVIT.FUSION.EARLY_FUSION_TYPE == 'kv':
                pass

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines

        if up_rate is not None and up_rate > 1:  # ?
            mlp_dim_out = dim * up_rate
            pdb.set_trace()
        else:
            mlp_dim_out = dim_out 

        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        ) 

        if layer_scale_init_value > 0: 
            self.gamma_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim_out)),
                requires_grad=True,
            )
        else:
            self.gamma_1, self.gamma_2 = None, None  #

        if dim != dim_out: 
            self.proj = nn.Linear(dim, dim_out) 

        self.pool_skip = (
            nn.MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(stride_skip) > 0
            and numpy.prod(stride_skip) > 1
            and len(kernel_q) != 0
            and len(kernel_kv) != 0 
            else None
        ) 

        ## check the kernel;
        if self.pool_skip != None: 
            assert kernel_skip[0] == 1 # 
            assert stride_skip[0] == 1 #
        
        self.attn_memories = None
    
    def empty_cache(self):
    
        self.attn_memories = None
    
        self.attn.q_cache = torch.empty(0)
        self.attn.k_cache = torch.empty(0)
        self.attn.v_cache = torch.empty(0)
        self.attn.x_cache = torch.empty(0)

    def forward(self, x, long_inputs=None, thw_shape=None, long_memory_mask=None):
        x_norm = self.norm1(x)  # ([2, 25089, 96])
        
        if self.cfg.MODEL.LONG_MEMORY_ENABLE:
            
            if self.early_fusion and self.cfg.MVIT.FUSION.EARLY_FUSION_TYPE == 'kv':
                               
                long_inputs = einops.rearrange(long_inputs, 'b t hw c -> b (t hw) c')
                long_inputs = self.norm1(long_inputs)    
                
                x_block, thw_shape_new = self.attn( # long_inputs.shape 2 8 49 384; 
                    x_norm, thw_shape, long_inputs, long_memory_mask)  # tensor, temporal spatial dimension;
            
            else:
                x_block, thw_shape_new = self.attn(
                    x_norm, thw_shape
                )  # tensor, tempor
        else:
            x_block, thw_shape_new = self.attn(
                x_norm, thw_shape
            )  # tensor, temporal spatial dimension;

        if (
            self.dim_mul_in_att and self.dim != self.dim_out
        ): 
            x = self.proj(x_norm)

        x_res, _ = attention_pool(
            x,
            self.pool_skip,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            spatial_temporal=self.st,
        ) 

        if self.gamma_1 is not None:
            x = x_res + self.drop_path(self.gamma_1 * x_block)  #
        else:
            x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x) 
        x_mlp = self.mlp(x_norm)  # FFN
        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm) 
        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * x_mlp)
        else:
            x = x + self.drop_path(x_mlp)

        if self.cfg.MODEL.LONG_MEMORY_ENABLE:
            if self.early_fusion:
                if self.cfg.MVIT.FUSION.EARLY_FUSION_TYPE == 'ca':
                    # x = einops.rearrange(x, 'b (t hw) c -> b t hw c', t = thw_shape[0])
                    long_inputs = einops.rearrange(long_inputs, 'b t hw c-> b (t hw) c')

                    x = self.fusion_block(x, long_inputs,long_memory_mask)

        if thw_shape:
            return x, thw_shape_new 
        else:
            return x
    
    def stream_inference(self, x, long_inputs=None, thw_shape=None):
        
        x_norm = self.norm1(x)  # ([2, 25089, 96])
        if self.early_fusion and self.cfg.MVIT.FUSION.EARLY_FUSION_TYPE == 'kv':
            
            long_inputs = einops.rearrange(long_inputs, 'b t hw c -> b (t hw) c')
            
            long_inputs = self.norm1(long_inputs)
            x_block, thw_shape_new = self.attn.stream_inference( # long_inputs.shape 2 8 49 384; 
                x_norm, thw_shape, long_inputs
            )  # tensor, temporal spatial dimension;
        
        else:
            x_block, thw_shape_new = self.attn.stream_inference(
                x_norm, thw_shape
            )  # tensor, temporal spatial dimension;

        if (
            self.dim_mul_in_att and self.dim != self.dim_out
        ): 
            x = self.proj(x_norm)

        x_res, _ = attention_pool(
            x,
            self.pool_skip,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            spatial_temporal=self.st,
        ) 

        if self.gamma_1 is not None:
            x = x_res + self.drop_path(self.gamma_1 * x_block)  #
        else:
            x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x) 
        x_mlp = self.mlp(x_norm)  # FFN
        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm) 
        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * x_mlp)
        else:
            x = x + self.drop_path(x_mlp)

        if self.cfg.MODEL.LONG_MEMORY_ENABLE:
            if self.early_fusion:

                if self.cfg.MVIT.FUSION.EARLY_FUSION_TYPE == 'ca':
                    # x = einops.rearrange(x, 'b (t hw) c -> b t hw c', t = thw_shape[0])
                    long_inputs = einops.rearrange(long_inputs, 'b t hw c-> b (t hw) c')
                    x = self.fusion_block(x, long_inputs)

        if thw_shape:
            return x, thw_shape_new  
        else:
            return x
