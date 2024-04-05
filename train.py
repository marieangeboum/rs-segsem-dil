import os
import glob
import time

import fnmatch
import random
import logging
import datetime

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
    parser.add_argument("--initial_lr", type=float, default = 0.01)
    parser.add_argument("--final_lr", type=float, default = 0.005)
    parser.add_argument("--lr_milestones", nargs=2, type=float, default=(20,80))
    parser.add_argument("--epoch_len", type=int, default=10000)
    parser.add_argument("--sup_batch_size", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument('--img_aug', type=str, default='d4_rot90_rot270_rot180_d1flip')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--train_split_coef', type = float, default = 0.75)   
    parser.add_argument('--sequence_path', type = str, default = "sequence_{}/")
    parser.add_argument('--strategy', type = str, default = 'FT_lora')
    parser.add_argument('--buffer_size', type = float, default = 0.2)
    parser.add_argument('--config_file', type = str, 
                        default = "/d/maboum/rs-segsem-dil/model/configs/config.yml")
    args = parser.parse_args()
        
    config_file = args.config_file
    config = load_config_yaml(file_path = config_file)
    # Get current date and time
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
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
    class_names = data_config["classnames_binary"]
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
    
    wandb.login(key ="ad58a41a99168fb44b86a70954b3728fe7818df2")
    

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True) 
    random.seed(seed)
    
    logger = logging.getLogger(__name__)
    logfile = os.path.join(config["checkpoints"],
                           f'run_{formatted_datetime}_{args.strategy}.log')
    logging.basicConfig(filename=f'run_{formatted_datetime}_{args.strategy}.log', 
                        level=logging.DEBUG, filemode = 'w', force=True)
    logging.info(f"{args} \n\n")
    logging.info(f"hyperparameters for data processing : {data_config} \n\n")
    logging.info(f"model hyperparams: {model_config} \n\n")
    logging.info(f"paramètres lora : {lora_params} \n\n")

    list_of_tuples = [(item, data_sequence.index(item)) for item in data_sequence]
    if not os.path.exists(args.sequence_path.format(seed)):
        os.makedirs(os.path.join(config["checkpoints"],
                                 args.sequence_path.format(seed)))  
        
    train_imgs = []
    test_imgs = []
    
    
    segmodel = Segmenter(im_size, n_layers, d_model, 4*d_model, n_heads,n_class,
                         patch_size, selected_model, lora_rank, lora_alpha).to(device)
    segmodel.load_pretrained_weights()

    if train_type == "lora":
        segmodel.apply_lora(lora_rank, lora_alpha, n_class)
        segmodel.to(device)
        num_params = sum(p.numel() for p in segmodel.parameters() if p.requires_grad)
        logging.info(f"training strategy: {train_type}M \n\n")
        logging.info(f"trainable parameters: {num_params/2**20:.4f}M \n\n")
    elif train_type == "finetuning":
        num_params = sum(p.numel() for p in segmodel.parameters() if p.requires_grad)
        logging.info(f"training strategy: {train_type}M \n\n")
        logging.info(f"trainable parameters: {num_params/2**20:.4f}M \n\n")

    test_dataloaders = []
    for step,domain in enumerate(data_sequence):
        # Définition de modèles
        model_path = os.path.join(config["checkpoints"],args.sequence_path.format(seed), 
                                  '{}_{}_{}'.format(args.strategy,seed, step)) 
        
        if step > 0 : 
            pretrained_segmodel = torch.load(os.path.join(config["checkpoints"],args.sequence_path.format(seed), 
                                      '{}_{}_{}'.format(args.strategy,seed, step-1)))
            segmodel.load_state_dict(pretrained_segmodel)
       
        wandb.init(project="baseline-experiments",
                    tags = str(step), 
                    name = str(step)+'_'+args.strategy+'_'+str(seed), 
                    config = data_config) 
          
        img = glob.glob(os.path.join(directory_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))
        random.shuffle(img)
        train_imgs += img[:int(len(img)*args.train_split_coef)]
        test_imgs  += img[int(len(img)*args.train_split_coef):]
        
        # if args.strategy == 'FT':
        domain_img = [item for item in train_imgs if  
                    fnmatch.fnmatch(item, os.path.join(directory_path, 
                    '{}/Z*_*/img/IMG_*.tif'.format(domain)))]
        random.shuffle(domain_img)
        
        domain_img_test = [item for item in test_imgs if  
                    fnmatch.fnmatch(item, os.path.join(directory_path, 
                    '{}/Z*_*/img/IMG_*.tif'.format(domain)))]
        # Train&Validation dataset
        domain_img_train = domain_img[:int(len(domain_img)*args.train_split_coef)]
        domain_img_val = domain_img[int(len(domain_img)*args.train_split_coef):]
        
        train_dataloader = create_train_dataloader(domain_img_train, directory_path, 
                                                   im_size, win_size, win_stride, 
                                                   args.img_aug, args.workers, 
                                                   args.sup_batch_size, args.epoch_len)
        
        val_dataloader = create_val_dataloader(domain_img_val, directory_path, 
                                               im_size, win_size, win_stride, 
                                               args.img_aug, args.workers, 
                                               args.sup_batch_size, args.epoch_len)
       
        test_dataloaders.append(create_test_dataloader(domain_img_test, directory_path, 
                                                 im_size, win_size, win_stride, 
                                                 args.img_aug, args.workers, 
                                                 args.sup_batch_size, args.epoch_len))
        # Callbacks 
        early_stopping = EarlyStopping(patience=20, verbose=True,  delta=0.001,path=model_path)
        optimizer = SGD(segmodel.parameters(),
                        lr=args.initial_lr,
                        momentum=0.9)
        loss_fn= torch.nn.BCEWithLogitsLoss().cuda() 
        scheduler = LambdaLR(optimizer,lr_lambda= lambda_lr, verbose = True)
        accuracy = Accuracy(task='binary',num_classes=n_class).cuda()
        # accuracy = Accuracy(num_classes=n_class).cuda()
        for epoch in range(1,args.max_epochs):

            time_ep = time.time() 
            segmodel,train_loss, train_acc = train_function(segmodel,train_dataloader, n_channels, 
                                                          device,optimizer, loss_fn, accuracy ,scheduler)
            print(train_loss, train_acc)
            segmodel,val_acc, val_loss, metrics_df, confusion_mat = validation_function(segmodel,val_dataloader, 
                                                                                        n_channels, device,optimizer, 
                                                                                        loss_fn, accuracy ,scheduler, 
                                                                                        class_names, eval_freq)
            early_stopping(val_loss['loss'],segmodel)
            wandb.log({'Metrics Class': wandb.Table(dataframe= metrics_df)})
            if early_stopping.early_stop :
                break
            wandb.log({"val_accuracy":val_acc['acc'], 
                       "val_loss": val_loss['loss'], 
                       "train_accuracy": train_acc["acc"], 
                       "train_loss": train_loss["loss"], 
                       "epochs" : epoch+1, 
                       "time" : time_ep})
            
            time_ep = time.time() - time_ep
            
        # best_model = Segmenter(im_size, n_layers, d_model, 4*d_model, n_heads,
        #                        n_class, patch_size, selected_model, lora_rank, lora_alpha).to(device)
        # best_model = torch.load(model_path)
        for test_step, test_domain in enumerate(data_sequence[:step+1]) : 
            test_acc, metrics_df, confusion_mat = test_function(segmodel, 
                                                                test_dataloaders[test_step], 
                                                                n_channels, device, optimizer, 
                                                                loss_fn, accuracy, scheduler, 
                                                                class_names, eval_freq)
            wandb.log({"test_accuracy": test_acc["acc"]})
            logging.info(f"test_accuracy: {test_acc['acc']}M \n\n")
            wandb.log({f'Etape {step} : {test_domain}': wandb.Table(dataframe= metrics_df)})
            # wandb.log({"Predictions (Step {test_step} Domain {test_domain})": wandb_list})
        wandb.finish()

if __name__ == "__main__":
    main()   