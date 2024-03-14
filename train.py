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

from model.segmenter import Segmenter
from model.datasets import FlairDs

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
    parser.add_argument('--train_split_coef', type = float, default = 0.7)   
    parser.add_argument('--sequence_path', type = str, default = "sequence_{}/")
    parser.add_argument('--strategy', type = str, default = 'FT')
    parser.add_argument('--buffer_size', type = float, default = 0.2)
    args = parser.parse_args()

    config_file ="/d/maboum/rs-segsem-dil/model/configs/config.yml"
    config = load_config_yaml(file_path =  config_file)

    # domain_list =  os.listdir(directory_path)
    # selected_domains = select_random_domains(domain_list, num_domains_to_select = seq_length , seed = config["seed"])
    # print("Selected Domains:", selected_domains)
    # update_domain_sequence(config_file, selected_domains)
    # print("Domain sequence updated in", config_file)

    dataset = config["dataset"]
    data_config = dataset["flair1"]

    seed = config["seed"]

    directory_path = data_config["data_path"]
    print(directory_path)
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
    columns = ['run','step','ep', 'train_loss', 'train_acc','val_acc', 'val_loss','time', 'method']     
    print(columns)       
    try:
      torch.manual_seed(seed)
      torch.cuda.manual_seed(seed)
      torch.autograd.set_detect_anomaly(True) 

      list_of_tuples = [(item, data_sequence.index(item)) for item in data_sequence]
      print(list_of_tuples)
      if not os.path.exists(args.sequence_path.format(seed)):
          os.makedirs(args.sequence_path.format(seed))  
          
      train_imgs = []    
      test_imgs = []
      
      step = 0
      idx = step-1 if step !=0 else step
      for step,domain in enumerate(data_sequence):
          
          img = glob.glob(os.path.join(directory_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))
          print(os.path.join(directory_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))
          random.shuffle(img)
          train_imgs += img[:int(len(img)*args.train_split_coef)]
          test_imgs += img[int(len(img)*args.train_split_coef):]
          print(img)
          if args.strategy == 'FT':
              domain_img = [item for item in train_imgs if  fnmatch.fnmatch(item, os.path.join(directory_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))]
          random.shuffle(domain_img)
          # Train&Validation dataset
          domain_img_train = domain_img[:int(len(domain_img)*args.train_split_coef)]
          domain_img_val = domain_img[int(len(domain_img)*args.train_split_coef):]   
          train_dataloader = create_train_dataloader(domain_img_train, directory_path, im_size, win_size, win_stride, args.img_aug, args.num_workers, args.sup_batch_size, args.epoch_len)
          val_dataloader = create_val_dataloader(domain_img_val, directory_path, im_size, win_size, win_stride, args.img_aug, args.num_workers, args.sup_batch_size, args.epoch_len)

          # Définition de modèles
          model_path= os.path.join(args.sequence_path.format(seed), '{}_{}_{}'.format(args.strategy,seed, step)) 
          print(model_path)
          segmodel = Segmenter(in_channels= n_channels, scale=0.05, patch_size=16, image_size=256, 
                            enc_depth=12, dec_depth=6, enc_embdd=768, dec_embdd=768, n_cls=n_class)
          # Callbacks 
          early_stopping = EarlyStopping(patience=20, verbose=True,  delta=0.001,path=model_path)
         
          print(segmodel)
          optimizer = SGD(
                segmodel.parameters(),
                lr=args.initial_lr,     
                momentum=0.9)
          loss_fn = torch.nn.BCEWithLogitsLoss() 
          scheduler = LambdaLR(optimizer,lr_lambda= lambda_lr, verbose = True)
          accuracy = Accuracy(num_classes=n_class).cuda()
          model = train_function(segmodel, epochs, train_dataloader, val_dataloader, n_channels, device, optimizer, loss_fn, accuracy, scheduler, args.strategy, step)       
    except Exception as e:
        # Handle the exception    
        error_message = f"An error occurred: {str(e)}"
        # error_trace = traceback.format_exc()  # Get the formatted stack trace
        # logging.error("%s\n%s", error_message, error_trace)
        # # Write the error message to a text file        
        # print(error_message, error_trace)

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