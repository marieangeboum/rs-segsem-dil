import torch
import copy
import timm

import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns

from model.vit import *
# from model.vit import _create_vision_transformer
from model.decoder import *

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, checkpoint_seq
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model


class Segmenter(nn.Module):
    def __init__(self,
                 in_channels,
                 scale,
                 patch_size,
                 image_size,
                 enc_depth,
                 # dec_depth,
                 # variant,
                 enc_embdd,
                 # dec_embdd,
                 n_cls):
        super().__init__()
        self.encoder = ViT(in_channels,
                            patch_size,
                            enc_embdd,
                            image_size,
                            enc_depth)
        # model_kwargs = dict(patch_size=patch_size, embed_dim=enc_embdd, depth=enc_depth, num_heads=enc_depth)
        # self.encoder = _create_vision_transformer(variant, pretrained=True, **model_kwargs)
       
        self.decoder = DecoderLinear(n_cls, patch_size, embedd_dim=enc_embdd)

    def forward(self, img):
        H, W = img.size(2), img.size(3)
        x = self.encoder(img)
        x = x[:, 1:]  # remove Cls token
        # x = x['fmaps']
        masks = self.decoder(x, (H, W))
        out = F.interpolate(masks, size=(H, W), mode="bilinear")
        return out

# model=Segmenter(3,0.05,16,256,12,6,768,768,1)
# print(model(torch.randn([16,3,256,256])).shape)
