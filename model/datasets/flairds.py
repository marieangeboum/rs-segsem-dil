import os
import sys
from torch.utils.data import Dataset
import torch
from dl_toolbox.utils import get_tiles
from dl_toolbox.torch_datasets.utils import *
from dl_toolbox.torch_datasets.commons import minmax

import rasterio
import imagesize
import numpy as np
from rasterio.windows import Window, bounds, from_bounds, shape
from rasterio.plot import show

from dl_toolbox.utils import MergeLabels, OneHot
import matplotlib.pyplot as plt

class FlairDs(Dataset):

    def __init__(self,image_path,tile,fixed_crops, crop_size,crop_step, img_aug,label_path=None, *args,**kwargs):

        self.image_path = image_path # path to image
        self.label_path = label_path # pth to corresponding label
        self.tile = tile # initializing a tile to be extracted from image
        self.crop_windows = list(get_tiles(
            nols=tile.width,
            nrows=tile.height, 
            size=crop_size,
            step=crop_step,
            row_offset=tile.row_off,
            col_offset=tile.col_off)) if fixed_crops else None # returns a list of tile these are crop extracted from the initial img
        #print("crop windows", self.crop_windows)
        self.crop_size = crop_size # initializing crop size
        self.img_aug = get_transforms(img_aug)


    def read_label(self, label_path, window):
        pass

    def read_image(self, image_path, window):
        pass

    def __len__(self):
        # returns nb of cropped windows
        return len(self.crop_windows) if self.crop_windows else 1

    def __getitem__(self, idx):
        ''' Given index idx this function loads a sample  from the dataset based on index.
            It identifies image'location on disk, converts it to a tensor
        '''

        if self.crop_windows:# if self.windows is initialized correctly begin iteration on said list
            window = self.crop_windows[idx]

        else: # otherwise use Winodw method from rasterio module to initilize a window of size cx and cy
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1) # why add those randint ?
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)

        #image = self.read_image(image_path=self.image_path,window=window)  # class inheritance

        ## Not here --> # vizualise the window crops extracted from the input image

        with rasterio.open(self.image_path, 'r') as image_file:
            image_rasterio = image_file.read(window=window, out_dtype=np.float32) # read the cropped part of the image
            img_path_strings = self.image_path.split('/')
            domain_pattern = img_path_strings[-4]

        # converts image crop into a tensor more precisely returns a contiguous tensor containing the same data as self tensor.
        image = torch.from_numpy(image_rasterio).float().contiguous()
        

        label = None
        if self.label_path:
            label = self.read_label(
                label_path=self.label_path,
                window=window)
            ## Not here --> # vizualise the window crops extracted from the input image

            with rasterio.open(self.label_path, 'r') as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)
                #print ('label', type(label))
                #show(label)
            # converts label crop into contiguous tensor
            
            label = torch.from_numpy(label).float().contiguous()
            
            bati_label_1 = (torch.eq(label, 1))
            bati_label_18 = (torch.eq(label, 18))
            final_mask_bati = (bati_label_1).float()
            # print(bati_label)

        if self.img_aug is not None:            
            
            #final_image, final_mask = self.img_aug(img = image_rasterio)
                 
                        
            final_image, final_mask = self.img_aug(img=image, label=final_mask_bati)
            
        else:
            final_image, final_mask = image, final_mask_bati
        
        return {'orig_image':image,
                'orig_mask': label,
                'id' : domain_pattern,
                'window':window,
                'image':final_image,
                'mask':final_mask}

class ClassifFlairDs(Dataset):

    def __init__(self,image_path,tile,fixed_crops, crop_size,crop_step, img_aug,label_path=None, *args,**kwargs):

        self.image_path = image_path # path to image
        self.label_path = label_path # pth to corresponding label
        self.tile = tile # initializing a tile to be extracted from image
        self.crop_windows = list(get_tiles(
            nols=tile.width,
            nrows=tile.height, 
            size=crop_size,
            step=crop_step,
            row_offset=tile.row_off,
            col_offset=tile.col_off)) if fixed_crops else None # returns a list of tile these are crop extracted from the initial img
        #print("crop windows", self.crop_windows)
        self.crop_size = crop_size # initializing crop size
        self.img_aug = get_transforms(img_aug)


    def read_label(self, label_path, window):
        pass

    def read_image(self, image_path, window):
        pass

    def __len__(self):
        # returns nb of cropped windows
        return len(self.crop_windows) if self.crop_windows else 1

    def __getitem__(self, idx):
        ''' Given index idx this function loads a sample  from the dataset based on index.
            It identifies image'location on disk, converts it to a tensor
        '''

        if self.crop_windows:# if self.windows is initialized correctly begin iteration on said list
            window = self.crop_windows[idx]

        else: # otherwise use Winodw method from rasterio module to initilize a window of size cx and cy
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1) # why add those randint ?
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)

        #image = self.read_image(image_path=self.image_path,window=window)  # class inheritance

        ## Not here --> # vizualise the window crops extracted from the input image

        with rasterio.open(self.image_path, 'r') as image_file:
            image_rasterio = image_file.read(window=window, out_dtype=np.float32) # read the cropped part of the image
            img_path_strings = self.image_path.split('/')
            domain_pattern = img_path_strings[-4]

        # converts image crop into a tensor more precisely returns a contiguous tensor containing the same data as self tensor.
        image = torch.from_numpy(image_rasterio).float().contiguous()
        

        label = None
        if self.label_path:
            label = self.read_label(
                label_path=self.label_path,
                window=window)
            ## Not here --> # vizualise the window crops extracted from the input image

            with rasterio.open(self.label_path, 'r') as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)
                #print ('label', type(label))
                #show(label)
            # converts label crop into contiguous tensor
            
            label = torch.from_numpy(label).float().contiguous()
            label_classification = torch.unique(label)
            
            bati_label_1 = (torch.eq(label, 1))
            bati_label_18 = (torch.eq(label, 18))
            final_mask_bati = (bati_label_1).float()
            final_label_classification = torch.unique(bati_label_1)
            # print(bati_label)

        if self.img_aug is not None:            
            
            #final_image, final_mask = self.img_aug(img = image_rasterio)
                 
                        
            final_image, final_mask = self.img_aug(img=image, label=final_mask_bati)
            
        else:
            final_image, final_mask = image, final_mask_bati
        
        return {'orig_image':image,
                'orig_label': label_classification,
                'id' : domain_pattern,
                'window':window,
                'image':final_image,
                'label':final_label_classification}

def main():
    dataset = FlairDs(
        image_path=os.path.join('data/flair_merged/train/', 'D017_2018/IMG_Z2_AU.tif'),
        label_path=os.path.join('data/flair_merged/train/', 'D017_2018/MSK_Z2_AU.tif'),
        crop_size=256,
        crop_step=256,
        img_aug='imagenet',
        tile=Window(col_off=0, row_off=0, width=400, height=400),
        fixed_crops=True)
    print(len(dataset))
    for data in dataset:
        pass
    img = plt.imshow(dataset[0]['image'].numpy().transpose(1,2,0))

    plt.show()
# from PIL import Image
# tiff_file = image_path
# image=rasterio.open(tiff_file)

# array = image.read(1)
# array.shape



















