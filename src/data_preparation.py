import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
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
    ds_ratio: factor by which data is downsampled
    downsamp_type: what kind of downsampling to use
    data_shape: size of each volume
    num_volumes: number of volumes to be used for the dataset. The default value is to use all volumes availeable.
    """
    
    def __init__(self, dataset_path, device, num_slices = 3, ds_ratio = 2, downsamp_type = 'bicubic', data_shape = [240,240,155], num_volumes = np.inf, slices_per_volume = 1, concatenate_modalities = False, binary_mask = False, augment = True):
        
        self.dataset_path = Path(dataset_path)
        self.device = device
        self.num_slices = num_slices
        self.ds_ratio = ds_ratio
        self.data_shape = data_shape
        self.downsamp_type = downsamp_type
        self.output_dim = np.array([240,240,num_slices], dtype = np.int64)
        self.input_dim = np.array([240//ds_ratio,240//ds_ratio,num_slices], dtype = np.int64)
        self.augment = augment
        self.binary_mask = binary_mask
        self.concatenate_modalities = concatenate_modalities
        
        if slices_per_volume >= data_shape[2] - 2*(num_slices//2):
            self.slices_per_volume = data_shape[2] - 2*(num_slices//2)
        else:
            self.slices_per_volume = slices_per_volume
        
        
        subdir_list = [p for p in self.dataset_path.iterdir() if p.is_dir()]
        
        if(len(subdir_list) > num_volumes):
            subdir_list = subdir_list[:num_volumes]
        
        self.subdir_list = subdir_list
        self.num_volumes = len(subdir_list)
        
        #Total length of the dataset = number of 2.5D slices * number of volumes
        self.length = self.slices_per_volume * self.num_volumes
         
        
    def __len__(self):
        return self.length
    
    def zscore(self, data):
        non_zero_locs = data>0
        non_zero_data = data[non_zero_locs]
        
        if non_zero_data.size == 0:
            return data
        
        mean = np.mean(non_zero_data)
        std = np.std(non_zero_data)
        if std > 0:
            zscored_data =  (non_zero_data - mean) / std
        else:
            zscored_data = 0
            
        data[non_zero_locs] = zscored_data
        
        return data
        
    
    def downsize(self, img):
        #Downsize in each channel
        ds = 1/float(self.ds_ratio)
        if self.downsamp_type == 'bicubic':
            downscaled_image = ndimage.zoom(img, (1, ds, ds), order=3)
        if self.downsamp_type == 'bilinear':
            downscaled_image = ndimage.zoom(img, (1, ds, ds), order=1)

        return downscaled_image

    def __getitem__(self, idx):
        #Each idx will get one 2.5D slice of a particular volume
        volume_idx = idx // self.slices_per_volume
        slice_idx = idx % self.slices_per_volume
        
        if self.slices_per_volume == 1:
            slice_idx = self.data_shape[2]//2
            
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
        img_list = [self.zscore(img) if img.max()>0 else img for img in data_list]
    
        img_list = [img.transpose(2,0,1) for img in img_list] #Channel dim first
        
        
        #Apply augmentations
        if(self.augment):
            intensity_shift = np.random.uniform(-0.1, 0.1, size=(4,))
            intensity_scale = np.random.uniform(0.9, 1.1, size=(4,))
            axis_flip = np.random.binomial(n=1, p=0.5, size=(3,))

            for i, img in enumerate(img_list):
                img_list[i] += intensity_shift[i]
                img_list[i] *= intensity_scale[i]
                for j, ax in enumerate(axis_flip):
                    if ax: img_list[i] = np.flip(img_list[i], axis = j).copy()
                        
            for j, ax in enumerate(axis_flip[1:3]):
                if ax: mask = np.flip(mask, axis = j).copy()
                                       
        #Downsample each slice
        ds_img_list = [self.downsize(img) for img in img_list]
        class_list = [1,2,4]
        
       
        if self.binary_mask:
            mask[mask>=2] = 1
        else:
            full_mask = np.zeros((3,240,240), dtype = int)
            for i in range(0,3):
                temp_mask = np.zeros_like(mask)
                temp_mask[mask == class_list[i]] = 1
                full_mask[i,:,:] = temp_mask
            mask = full_mask
                
        img_list = [torch.from_numpy(img).to(self.device).to(torch.float32) for img in img_list]
        ds_img_list = [torch.from_numpy(img).to(self.device).to(torch.float32)  for img in ds_img_list]
        mask = torch.from_numpy(mask).to(self.device).to(torch.float32)
                

        if self.concatenate_modalities:
            concat_img = torch.zeros((self.num_slices * 4, self.output_dim[0], self.output_dim[1]))
            concat_ds_img = torch.zeros((self.num_slices * 4, self.input_dim[0], self.input_dim[1]))
            for i in range(4):
                concat_img[i*self.num_slices : (i+1)*self.num_slices, :, :] = img_list[i]
                concat_ds_img[i*self.num_slices : (i+1)*self.num_slices, :, :] = ds_img_list[i]
                
            return concat_img, concat_ds_img, mask
        
        #Image list contains 2.5D slices of: flair, t1, t1ce, t2 (in order)
        return img_list, ds_img_list, mask