"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn

from model.utils import init_weights, resize_pos_embed, Block

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _load_weights


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.distilled = distilled
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 2, d_model)
            )
            self.head_dist = nn.Linear(d_model, n_cls)
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 1, d_model)
            )

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_features:
            return x

        if self.distilled:
            x, x_dist = x[:, 0], x[:, 1]
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            x = (x + x_dist) / 2
        else:
            x = x[:, 0]
            x = self.head(x)
        return x

    def get_attention_map(self, im, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as tt

# import matplotlib.pyplot as plt

# import os
# import torch
# import torchvision

# from PIL import Image
# from einops import repeat
# from einops import rearrange
# from einops.layers.torch import Rearrange, Reduce

# from torchsummary import summary
# from torchvision.utils import make_grid
# from torchvision.datasets import ImageFolder
# from torchvision.transforms.functional import to_pil_image
# from torch.utils.data import DataLoader

# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size=256):
#         self.patch_size = patch_size
#         super().__init__()
#         self.projection = nn.Sequential(
#             nn.Conv2d(in_channels, emb_size,
#                       kernel_size=patch_size, stride=patch_size),
#             Rearrange('b e (h) (w) -> b (h w) e')
#         )
#         self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
#         self.position = nn.Parameter(torch.randn(
#             (img_size//patch_size)**2+1, emb_size))

#     def forward(self, x):
#         b, _, _, _ = x.shape
#         x = self.projection(x)
#         # cls token added  x batch times and appended
#         cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
#         x = torch.cat([cls_tokens, x], dim=1)
#         x += self.position
#         return x


# class MultiHeadAttention(nn.Module):
#     def __init__(self, emb_size: int = 768, num_heads: int = 12, dropout: float = 0.1):
#         super().__init__()
#         self.emb_size = emb_size
#         self.num_heads = num_heads
#         # fuse the queries, keys and values in one matrix

#         self.qkv = nn.Linear(emb_size, emb_size * 3)
#         self.att_drop = nn.Dropout(dropout)
#         self.projection = nn.Linear(emb_size, emb_size)

#     def forward(self, x, mask=None):
#         # split keys, queries and values in num_heads
#         # 3 x batch x no_head x sequence_length x emb_size
#         qkv = rearrange(
#             self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
#         queries, keys, values = qkv[0], qkv[1], qkv[2]
#         # sum up over the last axis
#         # batch, num_heads, query_len, key_len
#         energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
#         if mask is not None:
#             fill_value = torch.finfo(torch.float32).min
#             energy.mask_fill(~mask, fill_value)

#         scaling = self.emb_size ** (1/2)
#         att = F.softmax(energy, dim=-1) / scaling
#         att = self.att_drop(att)
#         # sum up over the third axis
#         out = torch.einsum('bhal, bhlv -> bhav ', att, values)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         out = self.projection(out)
#         return out


# class ResidualBlock(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         res = x
#         x = self.fn(x, **kwargs)
#         x += res
#         return x


# class FeedForward(nn.Sequential):
#     def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.1):
#         super().__init__(
#             nn.Linear(emb_size, expansion * emb_size),
#             nn.GELU(),
#             nn.Dropout(drop_p),
#             nn.Linear(expansion * emb_size, emb_size),
#             nn.Dropout(drop_p),
#         )


# class TransformerEncoderBlock(nn.Sequential):
#     def __init__(self, emb_size=768, drop_p: float = 0.1, forward_expansion: int = 4, forward_drop_p: float = 0.1, ** kwargs):
#         super().__init__(
#             ResidualBlock(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 MultiHeadAttention(emb_size, **kwargs),
#                 nn.Dropout(drop_p)
#             )),
#             ResidualBlock(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 FeedForward(emb_size),
#                 nn.Dropout(drop_p)
#             )),
#         )

# class TransformerEncoder(nn.Sequential):
#     def __init__(self, depth: int = 6, **kwargs):
#         super().__init__(*[TransformerEncoderBlock(**kwargs)
#                            for _ in range(depth)])

# class ViT(nn.Sequential):
#     def __init__(self,
#                  in_channels: int = 3,
#                  patch_size: int = 16,
#                  emb_size: int = 768,
#                  img_size: int = 256,
#                  depth: int = 6,
#                  **kwargs):
#         super().__init__(
#             PatchEmbedding(in_channels, patch_size, emb_size, img_size),
#             TransformerEncoder(depth, emb_size=emb_size, **kwargs)
#         )



# # # model = ViT()
# # # print(model(torch.randn([1, 3, 32, 32])).shape)
