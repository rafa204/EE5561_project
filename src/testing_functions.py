from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.criterion import *

def test_model(model, test_loader, VAE):
    """
    Testing function for the model
    model: model to be tested
    test_loader: data loader for the test dataset
    VAE: boolean indicating whethere the VAE branch is present or not
    """
    
    model.eval()
    test_bar = tqdm(test_loader, desc=f"[Validation]")
    metrics = np.zeros((3,))
    
    with torch.no_grad():
        for img_list, ds_img_list, mask in test_bar:
            
            central_idx = img_list[0].shape[1]//2
            central_slice = img_list[0][:,central_idx,:,:].unsqueeze(1)
           
            seg_out, vae_out, mu, logvar = model(ds_img_list[0])

            metrics[0] += soft_dice_coeff(seg_out, mask).mean()
            metrics[1] += MSE_loss(vae_out, central_slice)
            metrics[2] += kl_loss(mu, logvar)
        
    metrics = metrics / len(test_bar)
    
    print(f"--- Validation results --- DICE: {metrics[0]:3f}, MSE: {metrics[1]:3f}, KL {metrics[2]:3f}")
    
    return metrics

def bin_mask_2_multi(mask):
    mask_temp = torch.round(mask)
    num_classes = mask.shape[0]
    multi_mask = torch.zeros((240, 240))
    for i in range(num_classes):
        multi_mask[mask_temp[i,:,:]==1] = i
    return multi_mask
    
    
def plot_examples(model, test_dataset, slices, save_path, vae = False):
    
    model.eval()
    print("Plotting results" + "-"*60)
    
    with torch.no_grad():
        j = 0
        for i in slices:
            
            img_list, ds_img_list, mask = test_dataset[i]
            
            mask = mask.unsqueeze(0)
            central_idx = img_list[0].shape[0]//2

            central_slice = img_list[0][central_idx,:,:].unsqueeze(0).unsqueeze(0)
            
            mask_multi = bin_mask_2_multi(mask.squeeze())
        

        if vae:
            
            seg_out, vae_out, mu, logvar = model(ds_img_list[0].unsqueeze(0))
            seg_out_multi = bin_mask_2_multi(seg_out.squeeze())
            
            dice_coeff = soft_dice_coeff(seg_out, mask).mean()
            mse_loss = MSE_loss(vae_out, central_slice)

            fig, ax = plt.subplots(2,2,figsize = (10,5))
            plt.gray()
            ax[0,0].imshow(central_slice.cpu().squeeze())
            ax[0,0].set_title("HR central slice")
            ax[0,1].imshow(mask_multi.cpu())
            ax[0,1].set_title("HR true mask")
            ax[1,0].imshow(seg_out_multi.cpu())
            ax[1,0].set_title(f"Predicted mask | Dice = {dice_coeff:.3f}")
            ax[1,1].imshow(vae_out.squeeze().cpu())
            ax[1,1].set_title(f"VAE output | MSE = {mse_loss:.3f}")
            
        else:
            
            seg_out = model(ds_img_list[0].unsqueeze(0))
            dice_coeff = soft_dice_coeff(seg_out, mask).mean()
        
            
            seg_out_multi = bin_mask_2_multi(seg_out.squeeze())
            loss = dice_loss(seg_out, mask)
            
            fig, ax = plt.subplots(1,3,figsize = (10,5))
            plt.gray()
            ax[0].imshow(central_slice.cpu().squeeze())
            ax[0].set_title("HR central slice")
            ax[1].imshow(mask_multi.cpu())
            ax[1].set_title("HR true mask")
            ax[2].imshow(seg_out_multi.cpu())
            ax[2].set_title(f"Predicted mask | Dice = {dice_coeff:.3f}")

            
        for ax in fig.get_axes():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='x', length=0)
            ax.tick_params(axis='y', length=0)
            
        fig.tight_layout()
        fig.show()
        fig.savefig(save_path / f"out_{j}.png")
        j += 1
        
    
    