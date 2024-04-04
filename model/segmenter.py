import torch
import copy
import timm

import torch.nn as nn
import torch.nn.functional as F


from model.vit import VisionTransformer
# from model.vit import _create_vision_transformer
from model.decoder import *

from model.vit import *
from model.utils import padding, unpadding
from timm.models.layers import trunc_normal_
from model.lora.lora import *
class Segmenter(nn.Module):
    def __init__(
        self,
        image_size,
        n_layers, 
        d_model, 
        d_ff,
        n_heads,
        n_cls,
        patch_size,
        variant,
        dropout = 0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.image_size = image_size,
        self.n_layers = n_layers,
        self.d_model = d_model,
        self.d_ff = d_ff, 
        self.n_heads = n_heads,
        self.variant = variant,
        self.dropout = 0.1,
        self.drop_path_rate = 0.0,
        self.drop_path_rate=0.0,
        self.distilled=False,
        self.channels=3
        
        self.encoder=VisionTransformer(
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
            )
        self.decoder = DecoderLinear(n_cls, patch_size, d_model)

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks
    def load_pretrained_weights(self):
        try : 
            timm_vision_transformer = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
            print("Pretrained model loaded successfully!")
            timm_vision_transformer.head = nn.Identity()
            weights = timm_vision_transformer.state_dict()
            self.encoder.load_state_dict(weights, False)
        except Exception as e:
            # Handle any exceptions that occur during loading
            print("An error occurred while loading the pretrained model:", e)
    
    def apply_lora(self, rank, alpha, n_cls):
        lora_vit = LoRA_ViT_timm(vit_model = self.encoder, 
                            r = rank, 
                            alpha = alpha, 
                            num_classes = n_cls)  
        self.encoder = lora_vit 
    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)