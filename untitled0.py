import os
import glob
import time

import fnmatch
import random
import logging
from rasterio.windows import Window
from argparse import ArgumentParser

import torch
import torch.nn as nn
import wandb

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from model.segmenter import Segmenter
from model.datasets import FlairDs
from torchmetrics import Accuracy
from model.configs.utils import *
from model.datasets.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = ArgumentParser()
parser.add_argument("--initial_lr", type=float, default = 0.001)
parser.add_argument("--final_lr", type=float, default = 0.0005)
parser.add_argument("--lr_milestones", nargs=2, type=float, default=(20,80))
parser.add_argument("--epoch_len", type=int, default=10000)
parser.add_argument("--sup_batch_size", type=int, default=2)
parser.add_argument("--crop_size", type=int, default=256)
parser.add_argument("--workers", default=6, type=int)
parser.add_argument('--img_aug', type=str, default='d4_rot90_rot180_d1flip_mixup')
parser.add_argument('--max_epochs', type=int, default=2)
parser.add_argument('--train_split_coef', type = float, default = 0.75)   
parser.add_argument('--sequence_path', type = str, default = "sequence_{}/")
parser.add_argument('--strategy', type = str, default = 'FT')
parser.add_argument('--buffer_size', type = float, default = 0.2)
parser.add_argument('--config_file', type = str, default = "/run/user/108646/gvfs/sftp:host=baymax/d/maboum/rs-segsem-dil/model/configs/config.yml")
args = parser.parse_args()

config_file = args.config_file
config = load_config_yaml(file_path = config_file)

dataset = config["dataset"]
data_config = dataset["flair1"]
seed = config["seed"]
directory_path = data_config["data_path"]
seq_length = data_config["seq_length"]
data_sequence = data_config["domains"]
epochs = data_config['epochs']
eval_freq = data_config['eval_freq']
im_size = data_config["im_size"]
lr = data_config['learning_rate']
win_size = data_config["window_size"]
win_stride = data_config["window_stride"]
n_channels = data_config['n_channels']
n_class = data_config["n_cls"]
class_names = data_config["classnames"]
eval_freq = data_config["eval_freq"]

selected_model = "vit_base_patch16_224"
model = config["model"]
model_config = model[selected_model]
im_size = model_config["image_size"]
patch_size = model_config["patch_size"]
d_model = model_config["d_model"]
n_heads = model_config["n_heads"]
n_layers = model_config["n_layers"]

train_type = config["train_type"]
lora_params = config["lora_parameters"]
lora_rank = lora_params["rank"]
lora_alpha = lora_params["rank"]

segmodel = Segmenter(im_size, n_layers, d_model, 4*d_model, n_heads,n_class, patch_size, selected_model, lora_rank, lora_alpha)
# from model.lora.lora import LoRA_ViT
# lora_vit = LoRA_ViT(vit_model = segmodel.encoder, r = lora_rank, alpha = lora_alpha, num_classes = n_class) 
segmodel.lora_finetuning(lora_rank, lora_alpha, n_class)