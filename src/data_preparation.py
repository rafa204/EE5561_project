import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import ndimage
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Define dataset for the training
class BRATS_dataset(Dataset):
    """
    Dataset class for the brain tumor segmentation data using a 2.5D slice setup
    
    INPUTS:
    dataset_path: path leading to the "MICCAI_BraTS2020_TrainingData" directory
    num_slices: number of slices in each 2.5D slab
    downsamp_ratio: factor by which data is downsampled
    downsamp_type: what kind of downsampling to use
    data_shape: size of each volume
    num_volumes: number of volumes to be used for the dataset. The default value is to use all volumes availeable.
    """
    
    def __init__(self, dataset_path, device, num_slices = 3, downsamp_ratio = 2, downsamp_type = 'bicubic', data_shape = [240, 240, 155], num_volumes = np.inf):
        
        self.dataset_path = Path(dataset_path)
        self.device = device
        self.num_slices = num_slices
        self.downsamp_ratio = downsamp_ratio
        self.data_shape = data_shape
        self.downsamp_type = downsamp_type
        self.slices_per_volume = data_shape[2] - 2*(num_slices//2)
        self.output_dim = np.array([240,240,num_slices], dtype = np.int64)
        self.input_dim = np.array([240//downsamp_ratio,240//downsamp_ratio,num_slices], dtype = np.int64)
        

        subdir_list = [p for p in self.dataset_path.iterdir() if p.is_dir()]
        
        if(len(subdir_list) > num_volumes):
            subdir_list = subdir_list[:num_volumes]
        
        self.subdir_list = subdir_list
        self.num_volumes = len(subdir_list)
        
        #Total length of the dataset = number of 2.5D slices * number of volumes
        self.length = self.slices_per_volume * self.num_volumes
        
    def __len__(self):
        return self.length
    
    def downsize(self, img):
        #Downsaize in each channel
        ds = 1/float(self.downsamp_ratio)
        if self.downsamp_type == 'bicubic':
            downscaled_image = ndimage.zoom(img, (1, ds, ds), order=3)
        if self.downsamp_type == 'bilinear':
            downscaled_image = ndimage.zoom(img, (1, ds, ds), order=1)

        return downscaled_image

    def __getitem__(self, idx):
        #Each idx will get one 2.5D slice of a particular volume
        volume_idx = idx // self.slices_per_volume
        slice_idx = idx % self.num_volumes
        slice_range = np.arange(slice_idx, slice_idx + self.num_slices)
        
        volume_path = self.subdir_list[volume_idx]
        file_list = [p for p in volume_path.iterdir() if p.is_file()]
        
        #List of all volumes
        vol_list = [nib.load(file_list[i]).get_fdata() for i in range(5)]
        
        #Get only the needed 2.5D slice
        data_list = [vol[:,:,slice_range] for vol in vol_list]
        
        #Get segmentation mask from the list and extract its central slice
        mask = data_list.pop(1)[:,:,self.num_slices//2].squeeze()
        
        #Normalize each 2.5D slice
        img_list = [img/img.max() if img.max()>0 else img for img in data_list]
        img_list = [img.transpose(2,0,1) for img in data_list] #Channel dim first
        
        #Downsample each slice
        ds_img_list = [self.downsize(img) for img in img_list]
        
        img_list = [torch.from_numpy(img).to(self.device).to(torch.float32) for img in img_list]
        ds_img_list = [torch.from_numpy(img).to(self.device).to(torch.float32)  for img in ds_img_list]
        mask = torch.from_numpy(mask).to(self.device).to(torch.float32)
        
        
        
        #Image list contains 2.5D slices of: flair, t1, t1ce, t2 (in order)
        return img_list, ds_img_list, mask