import timm
import torch
from lora import LoRA_ViT_timm
img = torch.randn(2, 3, 224, 224)
model = timm.create_model('vit_base_patch16_224', pretrained=True)
lora_vit = LoRA_ViT_timm(vit_model=model, r=4, alpha=4, num_classes=10)
pred = lora_vit(img)
print(pred.shape)