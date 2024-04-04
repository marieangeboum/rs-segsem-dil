import os
import glob
# import tqdm
import torch
import wandb
import numpy as np
import pandas as pd
import time as t
import fnmatch
import random
import logging
import tabulate
from tqdm import tqdm
import dl_toolbox.inference as dl_inf
from model.datasets import FlairDs
from rasterio.windows import Window
import torch.nn.functional as F
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
        img_path_strings = img_path.split('/')
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

def train_function(model,train_dataloader, n_channels, device,optimizer, loss_fn, accuracy ,scheduler):
    loss_sum = 0.0
    acc_sum = 0.0
    time_ep = t.time()
    for i, batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader)) :

        image = (batch['image'][:,:n_channels,:,:]/255.).to(device)
        target = (batch['mask']).to(device)
        torch.unique(target, dim=1)
        target = torch.squeeze(target, dim=1).long()
        optimizer.zero_grad()
        logits = model(image)
        loss = loss_fn(F.softmax(logits, dim=1), target)

        batch['preds'] = logits
        batch['image'] = image 
        
        loss.backward()
        optimizer.step()  

        acc_sum += accuracy(logits, target)
        loss_sum += loss.item()
    train_loss = {'loss': loss_sum / len(train_dataloader)}
    train_acc = {'acc': acc_sum/len(train_dataloader)}
    return model,train_loss, train_acc
    
def validation_function(model,val_dataloader, n_channels, device,optimizer, loss_fn, accuracy ,scheduler, class_labels, eval_freq):

    loss_sum = 0.0
    acc_sum = 0.0
    scheduler.step()
    for i, batch in tqdm(enumerate(val_dataloader), total = len(val_dataloader)):

        image = (batch['image'][:,:n_channels,:,:]/255.).to(device)
        target = (batch['mask']).to(device)  
        output = model(image)  
        target = torch.squeeze(target, dim=1).long()
        loss = loss_fn(F.softmax(output, dim=1), target)

    
        batch['preds'] = output
        batch['image'] = image 
        loss_sum += loss.item()
        # acc_sum += accuracy(torch.transpose(output,0,1).reshape(2, -1).t(), 
        #                     torch.transpose(target.to(torch.uint8),0,1).reshape(2, -1).t())
        acc_sum += accuracy(F.softmax(output, dim=1), target)
        confusion_mat = calculate_confusion_matrix(output, target, num_classes = 13)
        metrics_df = calculate_performance_metrics(confusion_mat, class_labels)
        wandb_image_list =[]
        # if i % eval_freq == 0 :
        wandb_image_list.append(
            wandb.Image(image[0,:,:,:].permute(1, 2, 0).cpu().numpy(), 
            masks={"prediction" :
            {"mask_data" : output.argmax(dim=1)[0,:,:].cpu().numpy(), "class_labels" : class_labels},
            "ground truth" : 
            {"mask_data" : target[0,:,:].cpu().numpy(), "class_labels" : class_labels}}, 
            caption= "{}_batch_{}".format(i, batch['id'])))
            
    val_loss = {'loss': loss_sum / len(val_dataloader)} 
    val_acc = {'acc': acc_sum/ len(val_dataloader)}
    return model,val_acc, val_loss, metrics_df, confusion_mat, wandb_image_list

def test_function(model,test_dataloader, n_channels, device,optimizer, loss_fn, accuracy ,scheduler, class_labels, eval_freq):

    loss_sum = 0.0
    acc_sum = 0.0
    scheduler.step()
    with torch.no_grad():
        
        for i, batch in tqdm(enumerate(test_dataloader), total = len(test_dataloader)):
    
            image = (batch['image'][:,:n_channels,:,:]/255.).to(device)
            target = (batch['mask']).to(device)  
            output = model(image)  
            target = torch.squeeze(target, dim=1).long()
            
            batch['preds'] = output
            batch['image'] = image 
            
            acc_sum += accuracy(F.softmax(output, dim=1), target)
            confusion_mat = calculate_confusion_matrix(output, target, num_classes = 13)
            metrics_df = calculate_performance_metrics(confusion_mat, class_labels)
            # wandb_image_list =[]
            # if i % eval_freq == 0 :
            #     wandb_image_list.append(
            #         wandb.Image(image[0,:,:,:].permute(1, 2, 0).cpu().numpy(), 
            #         masks={"prediction" :
            #         {"mask_data" : output.argmax(dim=1)[0,:,:].cpu().numpy(), "class_labels" : class_labels},
            #         "ground truth" : 
            #         {"mask_data" : target[0,:,:].cpu().numpy(), "class_labels" : class_labels}}, 
            #         caption= "{}_batch_{}".format(i, batch['id'])))

        test_acc = {'acc': acc_sum/ len(test_dataloader)}
    return test_acc, metrics_df, confusion_mat


def calculate_confusion_matrix(predictions, targets, num_classes):
   
    # Convertir les tensors PyTorch en tableaux NumPy
    predictions_np = predictions.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Initialiser la matrice de confusion
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Parcourir chaque image dans le batch
    for i in range(len(predictions_np)):
        # Comparer chaque pixel prédit avec le pixel de vérité terrain correspondant
        for true_class in range(num_classes):
            for predicted_class in range(num_classes):
                # Calculer le nombre de pixels prédits comme la classe 'predicted_class'
                # qui sont en fait de la classe 'true_class'
                confusion_matrix[true_class, predicted_class] += np.sum(
                    (predictions_np[i] == predicted_class) & (targets_np[i] == true_class)
                )

    return confusion_matrix

def calculate_performance_metrics(confusion_mat, classnames):
    
    num_classes = confusion_mat.shape[0]
    metrics = {'recall': [], 'precision': [], 'iou': [], 'f1': []}

    for i in range(num_classes):
        TP = confusion_mat[i, i]
        FN = np.sum(confusion_mat[i, :]) - TP
        FP = np.sum(confusion_mat[:, i]) - TP

        recall = TP / (TP + FN) if TP + FN != 0 else 0
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        iou = TP / (TP + FP + FN) if TP + FP + FN != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

        class_name = classnames[i]
        metrics['recall'].append((class_name, recall))
        metrics['precision'].append((class_name, precision))
        metrics['iou'].append((class_name, iou))
        metrics['f1'].append((class_name, f1))
        
        metrics_df = pd.DataFrame.from_dict({k: dict(v) for k, v in metrics.items()})
        metrics_df.index = metrics_df.index.map(class_names)

    return metrics_df

