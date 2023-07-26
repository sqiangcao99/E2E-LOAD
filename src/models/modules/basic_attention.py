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
    attn, q, k, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w, operator='self_attention'):
    """
    Decomposed Spatial Relative Positional Embeddings.
    """

    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape 

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
    )  # [B, H, q_t, qh, qw, k_h] # torch.Size([1, 4, 16, 14, 14, 2])

    # [B H q_t q_h q_w, dim] * q_w * k(Pooled) * dim->  [B, H, q_t, qh, qw, k_w]
    rel_w_q = torch.einsum(
        "bythwc,wkc->bythwk", r_q, Rw
    )  # [B, H, q_t, qh, qw, k_w] # torch.Size([1, 4, 16, 14, 14, 2])

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

    if operator == 'self_attention': 
        attn[:, :, :, 1:, :, 1:] = attn[:, :, :, 1:, :, 1:] + rel_h_q + rel_w_q 
    elif operator == 'cross_attention':
        attn[:, :, :, 1:, :, :] = attn[:, :, :, 1:, :, :] + rel_h_q + rel_w_q 
    else:
        raise NotImplementedError
    attn = einops.rearrange(
        attn, "b n q_t q_hw k_t k_hw -> b n (q_t q_hw) (k_t k_hw)"
    )

    return attn
    
def cal_rel_pos_temporal(
    attn, q, has_cls_embed, q_shape, k_shape, rel_pos_t, operator
):
    """
    Temporal Relative Positional Embeddings.
    """
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

    # Scale up rel pos if shapes for q and k are different.
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0) 

    dist_t = (
        torch.arange(q_t)[:, None] * q_t_ratio
        - torch.arange(k_t)[None, :] * k_t_ratio
    )

    dist_t += (k_t - 1) * k_t_ratio  
    Rt = rel_pos_t[dist_t.long()] 

    B, n_head, q_N, dim = q.shape  # torch.Size([1, 4, 3152, 96])
    r_q = einops.rearrange(q, "b n (t hw) c-> t (b n hw) c", t=q_shape[0])
    rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(
        0, 1
    )  # Rt.transpose(1, 2) # 788 16 16
    if operator == 'self_attention': 
        rel = einops.repeat(
            rel,
            "(b n q_hw) q_t k_t -> b n (q_t q_hw) (k_t k_hw)",
            b=B,
            n=n_head,
            q_hw=q_shape[1] * q_shape[2] + 1, 
            k_hw=k_shape[1] * k_shape[2] + 1,
        )
    elif operator == 'cross_attention':
        rel = einops.repeat(
            rel,
            "(b n q_hw) q_t k_t -> b n (q_t q_hw) (k_t k_hw)",
            b=B,
            n=n_head,
            q_hw=q_shape[1] * q_shape[2] + 1, 
            k_hw=k_shape[1] * k_shape[2],
        )
    else:
        raise NotImplementedError
    attn = attn + rel 
    return attn

class AttentionLayer(nn.Module):  # Pooling Attention, 
    def __init__(
        self,
        cfg,
        dim,
        kv_dim, 
        dim_out,
        q_size,
        kv_size, 
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        rel_pos_zero_init=False,
        separate_qkv=False,
        operator='cross_attention', 
    ): 

        super().__init__()
        self.separate_qkv = separate_qkv  # False
        self.dim = dim
        self.kv_dim = kv_dim
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads  
        self.scale = head_dim**-0.5  
        self.has_cls_embed = has_cls_embed  
        self.cfg = cfg

        self.q_size = q_size
        self.kv_size = kv_size
        self.operator = operator

        if separate_qkv:  
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(kv_dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(kv_dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)  #
        # if drop_rate > 0.0:
        #     self.proj_drop = nn.Dropout(drop_rate)
        if operator == 'cross_attention':
            self.rel_pos_spatial = cfg.MVIT.FUSION.INTERACTION.REL_POS_SPATIAL
            self.rel_pos_temporal = cfg.MVIT.FUSION.INTERACTION.REL_POS_TEMPORAL 
        elif operator == 'self_attention':
            self.rel_pos_spatial = cfg.MVIT.FUSION.ENHANCE.REL_POS_SPATIAL
            self.rel_pos_temporal = cfg.MVIT.FUSION.ENHANCE.REL_POS_TEMPORAL 
        else:
            raise NotImplementedError('Not Implemented yet.')

    
        if self.rel_pos_spatial: 
            assert q_size[1] == q_size[2] 
            assert kv_size[1] == kv_size[2] 

            rel_sp_dim = 2 * max(q_size[1], kv_size[1]) - 1  
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim)) 

            if not rel_pos_zero_init: 
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

        if self.rel_pos_temporal:
            rel_tp_dim = 2 * max(q_size[0], kv_size[0]) - 1

            self.rel_pos_t = nn.Parameter(torch.zeros(rel_tp_dim, head_dim))  #
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_t, std=0.02)
        
    def forward(self, query, key_value, long_memory_mask=None):
        B, N_q, C = query.shape
        B, N_kv, C = key_value.shape
        if not self.separate_qkv: 

            q_weight, kv_weight = self.qkv.weight[:self.dim], self.qkv.weight[self.dim:]
            q_bias, kv_bias = self.qkv.bias[:self.dim], self.qkv.bias[self.dim:]

            q = F.linear(query, q_weight, q_bias) 
            kv = F.linear(key_value, kv_weight, kv_bias)
            q = einops.rearrange(q, 'b thw (nh c) -> b nh thw c', nh=self.num_heads)
            kv = einops.rearrange(kv, 'b thw (nk nh c) -> nk b nh thw c', nk=2, nh=self.num_heads)
            k, v = kv[0], kv[1]
        else:  
            q = (
                self.q(query)
                .reshape(B, N_q, self.num_heads, -1)
                .permute(0, 2, 1, 3)  # 将
            )

            k = (
                self.k(key_value)
                .reshape(B, N_kv, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )
            v = (
                self.v(key_value)
                .reshape(B, N_kv, self.num_heads, -1)
                .permute(0, 2, 1, 3)
            )

        # Calculate the Attention; 
        attn = (q * self.scale) @ k.transpose(
            -2, -1
        )  # torch.Size([1, 8, 1600, 98]) # b nt (q_thw) kv_thw

        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                k,
                self.has_cls_embed,
                self.q_size,
                self.kv_size,
                self.rel_pos_h,  # [27, 96]
                self.rel_pos_w, 
                operator=self.operator
            )

        if self.rel_pos_temporal:
            attn = cal_rel_pos_temporal(
                attn,
                q,
                self.has_cls_embed,
                self.q_size,
                self.kv_size,
                self.rel_pos_t, 
                operator=self.operator
            )
        
        if self.cfg.MODEL.CASUAL_MASK_ENABLE:
            if self.operator == 'self_attention':
                casual_mask = generate_casual_mask(self.q_size[0], device=attn.device)
                casual_mask = einops.repeat(
                    casual_mask,
                    "q_t k_t -> b h (q_t q_hw) (k_t k_hw)",
                    q_hw=self.q_size[1] * self.q_size[2] + 1,
                    k_hw=self.kv_size[1] * self.kv_size[2] + 1,
                    b=1,
                    h=1,
                ) 
                attn = attn + casual_mask 
        
        if self.cfg.MODEL.LONG_MASK_ENABLE:
            if self.operator == 'cross_attention': 
                long_memory_mask = einops.repeat(
                    long_memory_mask,
                    'b k_t -> b h (q_t q_hw) (k_t k_hw)' ,
                    q_t=self.q_size[0], 
                    q_hw=self.q_size[1] * self.q_size[2] + 1,
                    k_hw=self.kv_size[1] * self.kv_size[2],
                    h=1,
                ) 
                attn = attn + long_memory_mask 
        
        attn = attn.softmax(dim=-1)
        x = attn @ v 
        
        if self.has_cls_embed:
            x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x) 
        return x


class AttentionBlock(nn.Module): 
    def __init__(
        self,
        cfg,
        dim,
        kv_dim, 
        dim_out,
        num_heads,
        q_size,
        kv_size,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        layer_scale_init_value=0.0, 
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        has_cls_embed=True,
        rel_pos_zero_init=False,
        dim_mul_in_att=False, 
        separate_qkv=False,
        operator='self_attention', 
    ):

        super().__init__()
        self.dim = dim
        self.kv_dim = kv_dim 
        self.dim_out = dim_out
        self.dim_mul_in_att = dim_mul_in_att 
        self.operator = operator
        self.cfg = cfg

        if self.operator == 'cross_attention' and self.cfg.MVIT.FUSION.INTERACTION.SPERATE_NORM:
            self.norm1_q = norm_layer(dim) 
            self.norm1_kv = norm_layer(kv_dim)  
        else:
            self.norm1 = norm_layer(dim) 
            assert kv_dim == dim
        
        att_dim = dim 
        self.attn = AttentionLayer(
            cfg,
            dim, 
            kv_dim, 
            att_dim, 
            num_heads=num_heads,
            q_size=q_size,
            kv_size=kv_size,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            norm_layer=norm_layer,
            has_cls_embed=has_cls_embed,
            separate_qkv=separate_qkv,
            operator=operator, 
        )
        
        self.drop_path = (DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity())
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed 
        mlp_dim_out = dim_out  

        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            # drop_rate=drop_rate,
        )  

        if layer_scale_init_value > 0: 
            self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim_out)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None 
        
        if dim != dim_out: 
            self.proj = nn.Linear(dim, dim_out)  

    def forward(self, query, key, long_memory_mask=None):
        
        if self.operator == 'cross_attention' and self.cfg.MVIT.FUSION.INTERACTION.SPERATE_NORM:
            query_norm = self.norm1_q(query)
            key_norm = self.norm1_kv(key) 
        else:
            query_norm = self.norm1(query)
            key_norm = self.norm1(key) 
        
        query_block = self.attn(query_norm, key_norm, long_memory_mask) # support ca and sa; 

        if (self.dim_mul_in_att and self.dim != self.dim_out): 
            x = self.proj(x_norm)
            raise NotImplementedError('Not Implemented.')

        query_res = query 
        if self.gamma_1 is not None:
            query = query_res + self.drop_path(self.gamma_1 * query_block)  
        else:
            query = query_res + self.drop_path(query_block) 
        
        query_norm = self.norm2(query) # norm after short cut; 
        query_mlp = self.mlp(query_norm)  # FFN
        
        if not self.dim_mul_in_att and self.dim != self.dim_out:
            
            query = self.proj(query_norm)  
            raise NotImplementedError('Only Support Dim Multiply before Attntion Operations.')
        
        if self.gamma_2 is not None: 
            query = query + self.drop_path(self.gamma_2 * query_mlp) 
        else:
            query = query + self.drop_path(query_mlp) # details about the implementations; 

        return query