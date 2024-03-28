import os
import glob
import time
import tabulate
import fnmatch
import random
import seaborn as sns
import matplotlib.pyplot as plt
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
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--train_split_coef', type = float, default = 0.75)   
    parser.add_argument('--sequence_path', type = str, default = "sequence_{}/")
    parser.add_argument('--strategy', type = str, default = 'FT')
    parser.add_argument('--buffer_size', type = float, default = 0.2)
    args = parser.parse_args()
    config_file ="/d/maboum/rs-segsem-dil/model/configs/config.yml"
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
    
    wandb.login(key = "ad58a41a99168fb44b86a70954b3728fe7818df2")
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True) 

    list_of_tuples = [(item, data_sequence.index(item)) for item in data_sequence]
        
    train_imgs = []    
    test_imgs = []
    
    segmodel = Segmenter((im_size,im_size), n_layers, d_model, 4*d_model, 
                         n_heads, n_class, patch_size, selected_model).to(device)
    segmodel.load_pretrained_weights()

    test_dataloaders = []
    perf_dict = {}
    for step,domain in enumerate(data_sequence[:4]):
        
        model_path = os.path.join(config["checkpoints"],args.sequence_path.format(seed), 
                                  '{}_{}_{}'.format(args.strategy,seed, step)) 
        
        pretrained_segmodel = torch.load(os.path.join(config["checkpoints"],args.sequence_path.format(seed), 
                                    '{}_{}_{}'.format(args.strategy,seed, step)))
        segmodel.load_state_dict(pretrained_segmodel)

        wandb.init(project="domain-incremental-semantic-segmentation-flair1",
                        tags = "inference", 
                        name = "inf_{}_{}_{}".format(str(step),args.strategy,str(seed)), 
                        config = data_config)  
        
        img = glob.glob(os.path.join(directory_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))
        random.shuffle(img)
        test_imgs  += img[int(len(img)*args.train_split_coef):]
        
        random.shuffle(test_imgs)
        domain_img_test = [item for item in test_imgs if  
                    fnmatch.fnmatch(item, os.path.join(directory_path, 
                    '{}/Z*_*/img/IMG_*.tif'.format(domain)))]
        test_dataloaders.append(create_test_dataloader(domain_img_test, directory_path, 
                                                 im_size, win_size, win_stride, 
                                                 args.img_aug, args.workers, 
                                                 args.sup_batch_size, args.epoch_len))
        # Callbacks 
        optimizer = SGD(segmodel.parameters(),
                        lr=args.initial_lr,
                        momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss().cuda() 
        scheduler = LambdaLR(optimizer,lr_lambda= lambda_lr, verbose = True)
        accuracy = Accuracy(task='multiclass',num_classes=n_class).cuda()
        # accuracy = Accuracy(num_classes=n_class).cuda()
        perf_dict[domain] = {}
        for test_step, test_domain in enumerate(data_sequence[:step+1]) : 
            test_acc, metrics_df, confusion_mat = test_function(segmodel, 
                                                                test_dataloaders[test_step], 
                                                                n_channels, device, optimizer, 
                                                                loss_fn, accuracy, scheduler, 
                                                                class_names, eval_freq)
            perf_dict[domain][test_domain] = metrics_df.iou.replace(0, np.nan).dropna().mean()
            wandb.log({f"confusion_matrix_{test_step}_{test_domain}":wandb.Table(columns = list(class_names.keys()),data=confusion_mat)})
            wandb.log({f"test_accuracy_{test_step}_{test_domain}": test_acc["acc"]})
            wandb.log({f'Etape {step} : {test_domain}': wandb.Table(dataframe= metrics_df)})
            perf_df =  pd.DataFrame.from_dict(perf_dict, orient='index')
            perf_df_wandb = wandb.Table(dataframe=perf_df)
            wandb.log({f"Sequence Metrics" : perf_df_wandb})
        wandb.finish()

if __name__ == "__main__":
    main()   
