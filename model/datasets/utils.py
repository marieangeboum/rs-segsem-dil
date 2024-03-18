import os
import glob
import tqdm
import torch
import wandb
import time as t
import fnmatch
import random
import logging
import tabulate
from model.datasets import FlairDs
from rasterio.windows import Window
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.callbacks import *


def create_train_dataloader(domain_img_train, data_path, im_size, win_size, win_stride, 
                            img_aug, n_workers, sup_batch_size, epoch_len): 
    # Train dataset
    train_datasets = []
    for img_path in domain_img_train : 
        img_path_strings = img_path.split('/')
        
        domain_pattern = img_path_strings[-4]
        img_pattern = img_path_strings[-1].split('_')[-1].strip('.tif')
        lbl_path = glob.glob(os.path.join(data_path, '{}/Z*_*/msk/MSK_{}.tif'.format(domain_pattern, img_pattern)))[0]
        # print(img_path, domain_pattern,img_pattern,lbl_path)
        train_datasets.append(FlairDs(image_path = img_path, label_path = lbl_path, fixed_crops = False,
                            tile=Window(col_off=0, row_off=0, width=im_size, height=im_size),
                            crop_size=win_size,        
                            crop_step=win_stride,
                            img_aug=img_aug))
    print("train datasets", len(train_datasets))
    trainset = ConcatDataset(train_datasets) 
    train_sampler = RandomSampler(
        data_source=trainset,
        replacement=False,
        num_samples=epoch_len)
    train_dataloader = DataLoader(
        dataset=trainset,
        batch_size=sup_batch_size,
        sampler=train_sampler,
        collate_fn = CustomCollate(batch_aug = img_aug),
        num_workers=n_workers)  
    return train_dataloader
    
def create_val_dataloader(domain_img_val, data_path, im_size, win_size, win_stride, 
                            img_aug, n_workers, sup_batch_size, epoch_len): 
    val_datasets = []
    for img_path in domain_img_val : 
        print(img_path)
        img_path_strings = img_path.split('/')
        print(img_path_strings)
        domain_pattern = img_path_strings[-4]
        img_pattern = img_path_strings[-1].split('_')[-1].strip('.tif')
        lbl_path = glob.glob(os.path.join(data_path, '{}/Z*_*/msk/MSK_{}.tif'.format(domain_pattern, img_pattern)))[0]
        # print(img_path, domain_pattern,img_pattern,lbl_path)
        val_datasets.append(FlairDs(image_path = img_path, label_path = lbl_path, fixed_crops = True,
                            tile=Window(col_off=0, row_off=0, width=im_size, height=im_size),
                            crop_size=win_size,        
                            crop_step=win_stride,
                            img_aug=img_aug))
    print("val datasets", len(val_datasets))
    valset =  ConcatDataset(val_datasets)
    val_dataloader = DataLoader(
        dataset=valset,
        shuffle=False,
        batch_size=sup_batch_size,
        collate_fn = CustomCollate(batch_aug = img_aug),
        num_workers=n_workers)  
    return val_dataloader
    
def create_test_dataloader(domain_img_test, data_path, im_size, win_size, win_stride, 
                            img_aug, num_workers, sup_batch_size, epoch_len): 
    test_datasets = []
    for img_path in domain_img_test : 
        img_path_strings = img_path.split('/')
        domain_pattern = img_path_strings[-4]
        img_pattern = img_path_strings[-1].split('_')[-1].strip('.tif')
        lbl_path = glob.glob(os.path.join(data_path, '{}/Z*_*/msk/MSK_{}.tif'.format(domain_pattern, img_pattern)))[0]
        test_datasets.append(FlairDs(image_path = img_path, label_path = lbl_path, fixed_crops = True,
                            tile=Window(col_off=0, row_off=0, width=im_size, height=im_size),
                            crop_size=win_size,        
                            crop_step=win_stride,
                            img_aug=img_aug))
         
    testset =  ConcatDataset(test_datasets)
    test_dataloader = DataLoader(
        dataset=testset,
        shuffle=False,
        batch_size=sup_batch_size,
        collate_fn = CustomCollate(batch_aug = img_aug),
        num_workers=num_workers)
    return test_dataloader



def train_function(model, max_epochs, train_dataloader,val_dataloader, n_channels, device,optimizer, loss_fn, accuracy ,scheduler):
    
    columns = ['run','step','ep', 'train_loss', 'train_acc','val_acc', 'val_loss','time', 'method']
    start_epoch = 0 
    for epoch in range(max_epochs):
        loss_sum = 0.0
        acc_sum  = 0.0
        time_ep = t.time()
        
        for i, batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader)) :
           
            image = (batch['image'][:,:n_channels,:,:]/255.).to(device)
            target = (batch['mask']).to(device)                    
            
            optimizer.zero_grad()                         
            logits = model(image)
            loss = loss_fn(logits, target)

            batch['preds'] = logits
            batch['image'] = image 

            loss.backward()
            optimizer.step()  

            acc_sum += accuracy(torch.transpose(logits,0,1).reshape(2, -1).t(), 
                                torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
            loss_sum += loss.item()
      
        train_loss = {'loss': loss_sum / len(train_dataloader)}
        train_acc = {'acc': acc_sum/len(train_dataloader)}

        loss_sum = 0.0
        acc_sum = 0.0
        iou = 0.0
        precision = 0.0
        recall = 0.0  
        scheduler.step()
        # model.eval()
        for i, batch in tqdm(enumerate(val_dataloader), total = len(val_dataloader)):
           
            image = (batch['image'][:,:n_channels,:,:]/255.).to(device)
            target = (batch['mask']).to(device)  

            output = model(image)  
            loss = loss_fn(output, target)  

            batch['preds'] = output
            batch['image'] = image
            
            iou += metrics_per_class_df.IoU
            precision += metrics_per_class_df.Precision[1] 
            recall += metrics_per_class_df.Recall[1]
            
            loss_sum += loss.item()
            acc_sum += accuracy(torch.transpose(output,0,1).reshape(2, -1).t(), 
                                torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
            
            cm = compute_conf_mat(
                target.clone().detach().flatten().cpu(),
                ((torch.sigmoid(output)>0.5).cpu().long().flatten()).clone().detach(), 2)
            
            metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(cm.numpy()) 
                
        val_iou = {'iou':iou/len(val_dataloader)}
        val_precision = {'prec':precision/len(val_dataloader)}
        val_recall = {'recall':recall/len(val_dataloader)}
        val_loss = {'loss': loss_sum / len(val_dataloader)} 
        val_acc = {'acc': acc_sum/ len(val_dataloader)}
        
        early_stopping(val_loss['loss'],model)

        if early_stopping.early_stop:
                print("Early Stopping")
                break
            
        time_ep = t.time() - time_ep
        
        wandb.log({"val_accuracy":val_acc['acc'], 
                   "val_loss": val_loss['loss'], 
                   "val_iou": val_iou['iou'], 
                   "val_recall": val_recall['recall'], 
                   "val_precision": val_precision['prec'],
                   "train_accuracy": train_acc["acc"], 
                   "train_loss": train_loss["loss"], 
                   "epochs" : epoch+1, 
                   "time" : time_ep})

    return model

def save_table(table_name):
  table = wandb.Table(columns=['Original Image', 'Original Mask', 'Predicted Mask'], allow_mixed_types = True)
  for bx, data in tqdm(enumerate(val_dl), total = len(val_dl)):
    im, mask = data
    _mask = model(im)
    _, _mask = torch.max(_mask, dim=1)

    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(im[0].permute(1,2,0).detach().cpu()[:,:,0])
    plt.savefig("original_image.jpg")
    plt.close()

    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(mask.permute(1,2,0).detach().cpu()[:,:,0])
    plt.savefig("original_mask.jpg")
    plt.close()

    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(_mask.permute(1,2,0).detach().cpu()[:,:,0])
    plt.savefig("predicted_mask.jpg")
    plt.close()

    table.add_data(
        wandb.Image(cv2.cvtColor(cv2.imread("original_image.jpg"), cv2.COLOR_BGR2RGB)),
        wandb.Image(cv2.cvtColor(cv2.imread("original_mask.jpg"), cv2.COLOR_BGR2RGB)),
        wandb.Image(cv2.cvtColor(cv2.imread("predicted_mask.jpg"), cv2.COLOR_BGR2RGB))
    )

  wandb.log({table_name: table})




