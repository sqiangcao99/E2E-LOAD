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

    if spatial_temporal:
        
        if has_cls_embed: 
            tensor = einops.rearrange(
                tensor, "b n (t hw) c -> b n t hw c", t=thw_shape[0]
            ) 
            cls_tok, tensor = (
                tensor[:, :, :, :1, :],
                tensor[:, :, :, 1:, :],
            ) 

            B, N, T, L, C = tensor.shape 
            T, H, W = thw_shape # 16 14 14 
            # cls_tok torch.Size([1, 4, 16, 1, 96])
            # tensor torch.Size([1, 4, 16, 196, 96])
            assert T * H * W == L
            tensor = einops.rearrange(tensor, "b nh t (h w) c -> (b nh) c t h w", h=H, w=W)
        else:        
            B, N, L, C = tensor.shape  # torch.Size([2, 1, 25089, 96])
            T, H, W = thw_shape
            assert T * H * W == L
            tensor = einops.rearrange(tensor, 'b nh (t h w) c -> (b nh) c t h w', h=H, w=W) # 

    tensor = pool(tensor)
    # torch.Size([2, 96, 8, 56, 56])
    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]  
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]

    if spatial_temporal: 
        if has_cls_embed: 
            tensor = einops.rearrange(tensor, "(b n) c t h w -> b n t (h w) c", n=N)
            tensor = torch.cat(
                (cls_tok, tensor), dim=-2
            )  # torch.Size([1, 4, 16, 197, 96])
            tensor = einops.rearrange(tensor, "b n t hw c -> b n (t hw) c")
        else:
            tensor = einops.rearrange(tensor, '(b n) c t h w -> b n (t h w) c', n=N)

    if norm is not None:
        tensor = norm(tensor)
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
    Rh = rel_pos_h[dist_h.long()] # torch.Size([14, 7, 96])
    Rw = rel_pos_w[dist_w.long()]  # Apply indices selection;
  
    B, n_head, q_N, dim = q.shape  # torch.Size([2, 1, 25089, 96])
    
    if has_cls_embed: 
        r_q = einops.rearrange(q, "b n (t hw) c-> b n t hw c", t=q_shape[0])
        cls_tok, r_q = (
            r_q[:, :, :, :1, :],
            r_q[:, :, :, 1:, :],
        ) 
        r_q = einops.rearrange(r_q, "b n t (h w) c-> b n t h w c", h=q_shape[1]) # 
    else:
        r_q = einops.rearrange(q, "b n (t h w) c-> b n t h w c", t=q_shape[0], h = q_shape[1], w=q_shape[2])
        
    rel_h_q = torch.einsum(
        "bythwc,hkc->bythwk", r_q, Rh
    )  # [B, H, q_t, qh, qw, k_h] # torch.Size([1, 4, 16, 14, 14, 2])

    rel_w_q = torch.einsum(
        "bythwc,wkc->bythwk", r_q, Rw
    )  # [B, H, q_t, qh, qw, k_w] # torch.Size([1, 4, 16, 14, 14, 2])

    rel_h_q = rel_h_q[:, :, :, :, :, None, :, None].expand(
        -1, -1, -1, -1, -1, k_shape[0], -1, k_shape[2]
    )
    rel_w_q = rel_w_q[:, :, :, :, :, None, None, :].expand(
        -1, -1, -1, -1, -1, k_shape[0], k_shape[1], -1
    )

    if has_cls_embed:
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

        attn[:, :, :, 1:, :, 1:] = attn[:, :, :, 1:, :, 1:] + rel_h_q + rel_w_q

        attn = einops.rearrange(
            attn, "b n q_t q_hw k_t k_hw -> b n (q_t q_hw) (k_t k_hw)"
        )

    else:
        rel_h_q = einops.rearrange(
            rel_h_q, "b n q_t q_h q_w k_t k_h k_w-> b n (q_t q_h q_w) (k_t k_h k_w)"
        )
        rel_w_q = einops.rearrange(
            rel_w_q, "b n q_t q_h q_w k_t k_h k_w-> b n (q_t q_h q_w) (k_t k_h k_w)"
        )

        attn = attn + rel_h_q + rel_w_q

    return attn



def cal_rel_pos_temporal(
    attn, q, has_cls_embed, q_shape, k_shape, rel_pos_t
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
        raise NotImplementedError('Not Implement Error.')
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

    rel = einops.repeat(
        rel,
        "(b n q_hw) q_t k_t -> b n (q_t q_hw) (k_t k_hw)",
        b=B,
        n=n_head,
        q_hw=q_shape[1] * q_shape[2],
        k_hw=k_shape[1] * k_shape[2],
    )

    attn = attn + rel # only apply on the instance token; 

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
        # If True, perform pool before projection.
        pool_func=nn.Conv3d,
        pool_first=False,
        rel_pos_zero_init=False,
        residual_pooling=False,
        separate_qkv=False,
    ):
        super().__init__()
        self.pool_first = pool_first  # False
        self.separate_qkv = separate_qkv  # False
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads  
        self.scale = head_dim**-0.5  
        self.has_cls_embed = has_cls_embed  
        self.mode = mode  
        padding_q = [int(q // 2) for q in kernel_q] 
        padding_kv = [int(kv // 2) for kv in kernel_kv] 
        
        self.st = not (cfg.MVIT.SPATIAL.AGGREGATION.WORK_ENABLE)
        self.cfg = cfg
        
        if pool_first or separate_qkv: 
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        
        self.proj = nn.Linear(dim_out, dim_out) 

        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

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
            if pool_first:  #
                dim_conv = dim // num_heads if mode == "conv" else dim
            else:
                dim_conv = dim_out // num_heads if mode == "conv" else dim_out

            self.pool_q = (
                pool_func(
                    dim_conv,  #
                    dim_conv,  
                    kernel_q,
                    stride=stride_q,
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

        self.rel_pos_spatial = (not cfg.MVIT.SPATIAL.AGGREGATION.WORK_ENABLE)

        if self.rel_pos_spatial and self.cfg.MVIT.COMPRESSOR.REL_POS_SPATIAL:
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
 
        if self.cfg.MVIT.COMPRESSOR.REL_POS_TEMPORAL:
            self.rel_pos_t = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim)
            )  #
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_t, std=0.02)

        self.residual_pooling = residual_pooling # True; 

    def forward(self, x, thw_shape):

        B, N, _ = x.shape  # fix the shape here;

        if (
            self.pool_first # False; 
        ): 
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.num_heads

            x = x.reshape(B, N, fold_dim, -1).permute(
                0, 2, 1, 3
            ) 
            q = k = v = x 
        else: 
            assert self.mode != "conv_unshared"
            if not self.separate_qkv: 
                qkv = (
                    self.qkv(x)
                    .reshape(B, N, 3, self.num_heads, -1)
                    .permute(2, 0, 3, 1, 4) 
                )
                q, k, v = qkv[0], qkv[1], qkv[2] 
                # q.shape: ([2, 4, 32, 96])
            else: 
                q = k = v = x
                q = (
                    self.q(q)
                    .reshape(B, N, self.num_heads, -1)
                    .permute(0, 2, 1, 3)  # 将
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
            spatial_temporal=self.st
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

        attn = (q * self.scale) @ k.transpose(
            -2, -1
        ) 

        if self.rel_pos_spatial and self.cfg.MVIT.COMPRESSOR.REL_POS_SPATIAL:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                k,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h, 
                self.rel_pos_w, 
            )
        
        if self.cfg.MVIT.COMPRESSOR.REL_POS_TEMPORAL:
            attn = cal_rel_pos_temporal(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_t, 
            )

        attn = attn.softmax(dim=-1)
        x = attn @ v

        if self.residual_pooling: 
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
                raise NotImplementedError('Not open the cls_embed for compressor.')
            else:
                x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x) 

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
        pool_first=False,
        rel_pos_zero_init=False,
        residual_pooling=False,
        dim_mul_in_att=False,
        separate_qkv=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim) 
        self.dim_mul_in_att = dim_mul_in_att 

        self.st = not (cfg.MVIT.SPATIAL.AGGREGATION.WORK_ENABLE)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q] 
        stride_skip = stride_q 
        padding_skip = [int(skip // 2) for skip in kernel_skip] 

        att_dim = (dim_out if dim_mul_in_att else dim) 
        if (numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1): 
            raise NotImplementedError('Not implemented yet')
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = () 

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
            pool_first=pool_first,
            rel_pos_zero_init=rel_pos_zero_init, 
            residual_pooling=residual_pooling,
            separate_qkv=separate_qkv,
        )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines

        if up_rate is not None and up_rate > 1:  # ?
            mlp_dim_out = dim * up_rate
            raise NotImplementedError('compressor up_rates')
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
    
    def forward(self, x, thw_shape=None):

        x_norm = self.norm1(x)  # ([2, 25089, 96])
        
        x_block, thw_shape_new = self.attn(x_norm, thw_shape)  # tensor, temporal spatial dimension;

        if (self.dim_mul_in_att and self.dim != self.dim_out):  
            x = self.proj(x_norm)

        x_res, _ = attention_pool(
            x.unsqueeze(1),
            self.pool_skip,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            spatial_temporal=self.st, 
        ) 
        
        x_res = x_res.squeeze(1)
        
        if self.gamma_1 is not None:
            x = x_res + self.drop_path(self.gamma_1 * x_block)  #
        else:
            x = x_res + self.drop_path(x_block) # Pool+Attention, SC

        x_norm = self.norm2(x)  
        x_mlp = self.mlp(x_norm)  # FFN
        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)  
        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * x_mlp)
        else:
            x = x + self.drop_path(x_mlp) # FFN-SC

        if thw_shape:
            return x, thw_shape_new 
        else:
            return x 
