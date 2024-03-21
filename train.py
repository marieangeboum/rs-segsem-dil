import os
import glob
import time
import tabulate
import fnmatch
import random

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


def main(): 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser()
    parser.add_argument("--initial_lr", type=float, default = 0.001)
    parser.add_argument("--final_lr", type=float, default = 0.0005)
    parser.add_argument("--lr_milestones", nargs=2, type=float, default=(20,80))
    parser.add_argument("--epoch_len", type=int, default=10000)
    parser.add_argument("--sup_batch_size", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument('--img_aug', type=str, default='d4')
    parser.add_argument('--max_epochs', type=int, default=120)
    parser.add_argument('--train_split_coef', type = float, default = 0.75)   
    parser.add_argument('--sequence_path', type = str, default = "sequence_{}/")
    parser.add_argument('--strategy', type = str, default = 'FT')
    parser.add_argument('--buffer_size', type = float, default = 0.2)
    args = parser.parse_args()
    config_file ="/run/user/108646/gvfs/sftp:host=flexo/d/maboum/rs-segsem-dil/model/configs/config.yml"
    # config_file ="/d/maboum/rs-segsem-dil/model/configs/config.yml"
    config = load_config_yaml(file_path = config_file)
    
    # Learning rate 
    def lambda_lr(epoch):
    
        m = epoch / args.max_epochs
        if m < args.lr_milestones[0]:
            return 1
        elif m < args.lr_milestones[1]:
            return 1 + ((m - args.lr_milestones[0]) / (
                        args.lr_milestones[1] - args.lr_milestones[0])) * (
                               args.final_lr / args.initial_lr - 1)
        else:
            return args.final_lr / args.initial_lr
        
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

    selected_model = "vit_base_patch16_384"
    model = config["model"]
    model_config = model[selected_model]

    wandb.login(key = "a60322f26edccc6c3f79accc480d56e52e02750a")
    wandb.init(project="domain-incremental-semantic-segmentation-flair1")
   

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True) 

    list_of_tuples = [(item, data_sequence.index(item)) for item in data_sequence]
    if not os.path.exists(args.sequence_path.format(seed)):
        os.makedirs(args.sequence_path.format(seed))  
        
    train_imgs = []    
    test_imgs = []
    
    for step,domain in enumerate(data_sequence[:1]):
        wandb.init(tags = str(step), name = str(step)+'_'+args.strategy+'_'+str(seed), config = data_config)   
        img = glob.glob(os.path.join(directory_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))
        random.shuffle(img)
        train_imgs += img[:int(len(img)*args.train_split_coef)]
        test_imgs  += img[int(len(img)*args.train_split_coef):]
        
        # if args.strategy == 'FT':
        domain_img = [item for item in train_imgs if  
                    fnmatch.fnmatch(item, os.path.join(directory_path, 
                    '{}/Z*_*/img/IMG_*.tif'.format(domain)))]
        random.shuffle(domain_img)
        
        # Train&Validation dataset
        domain_img_train = domain_img[:int(len(domain_img)*args.train_split_coef)]
        domain_img_val = domain_img[int(len(domain_img)*args.train_split_coef):]
        train_dataloader = create_train_dataloader(domain_img_train, directory_path, im_size, 
                                                   win_size, win_stride, args.img_aug, args.workers, 
                                                   args.sup_batch_size, args.epoch_len)
        print("train_dataloader",len(train_dataloader))
        val_dataloader = create_val_dataloader(domain_img_val, directory_path, im_size, 
                                                win_size, win_stride, args.img_aug, 
                                                args.workers, args.sup_batch_size, args.epoch_len)
        # Définition de modèles
        model_path = os.path.join(args.sequence_path.format(seed), '{}_{}_{}'.format(args.strategy,seed, step)) 
        segmodel = Segmenter(in_channels= n_channels, scale=0.05, patch_size=16, image_size=256, 
                          enc_depth=model_config["n_layers"], enc_embdd=model_config["d_model"], n_cls=n_class).to(device)
        # segmodel = Segmenter(scale=0.05, patch_size= 16,enc_depth = model_config["n_layers"], 
                             # variant=selected_model,enc_embdd = model_config["d_model"], n_cls = n_class).to(device)
        # Callbacks 
        early_stopping = EarlyStopping(patience=20, verbose=True,  delta=0.001,path=model_path)
        optimizer = SGD(segmodel.parameters(),
                        lr=args.initial_lr,
                        momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss().cuda() 
        scheduler = LambdaLR(optimizer,lr_lambda= lambda_lr, verbose = True)
        # accuracy = Accuracy(task='multiclass',num_classes=n_class).cuda()
        accuracy = Accuracy(num_classes=n_class).cuda()
        for epoch in range(args.max_epochs):

            time_ep = t.time() 
            segmodel,train_loss, train_acc = train_function(segmodel,train_dataloader, n_channels, 
                                                          device,optimizer, loss_fn, accuracy ,scheduler)
            print(train_loss, train_acc)
            segmodel,val_acc, val_loss, per_class, macro_average, micro_average = validation_function(segmodel,val_dataloader, 
                                                                                        n_channels, device,optimizer, 
                                                                                        loss_fn, accuracy ,scheduler)
            early_stopping(val_loss['loss'],segmodel)
            wandb.log({'Metrics Class': wandb.Table(dataframe= per_class)})
            wandb.log({'Macro Average': wandb.Table(dataframe= macro_average)})
            wandb.log({'Micro Average': wandb.Table(dataframe= micro_average)})
            wandb.log({"val_accuracy":val_acc['acc'], 
                       "val_loss": val_loss['loss'], 
                       "train_accuracy": train_acc["acc"], 
                       "train_loss": train_loss["loss"], 
                       "epochs" : epoch+1, 
                       "time" : time_ep})
            if early_stopping.early_stop :
                break
            time_ep = t.time() - time_ep


if __name__ == "__main__":
    
    main()   


 # if step !=0 and args.strategy == 'ER':
 #     coef_replay = args.buffer_size/5
 #     past_domain_img = []
 #     # past_domain_lbl = []
 #     idx_past = 0 if step-5<0 else step-5
 #     for source_domain in data_sequence[idx_past:step]:
 #         a_domain_img = [item for item in train_imgs if fnmatch.fnmatch(item, os.path.join(directory_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))]
 #         # a_domain_lbl = [item for item in train_lbl if fnmatch.fnmatch(item, os.path.join(directory_path, '{}/Z*_*/msk/MSK_*.tif'.format(domain)))] 
 #         coef = int(len(a_domain_img)*coef_replay) if int(len(a_domain_img)*coef_replay)>0 else 1
 #         a_domain_img_train = a_domain_img[:coef]
 #         # a_domain_lbl_train = a_domain_lbl[:coef]
 #         past_domain_img += a_domain_img_train
 #         # past_domain_lbl += a_domain_lbl_train
 #     domain_img = [item for item in train_imgs if  fnmatch.fnmatch(item, os.path.join(directory_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))]
 #     # domain_lbl = [item for item in train_lbl if fnmatch.fnmatch(item, os.path.join(directory_path, '{}/Z*_*/msk/MSK_*.tif'.format(domain)))]
 #     domain_img = domain_img + past_domain_img
 #     # domain_lbl = domain_lbl + past_domain_lbl