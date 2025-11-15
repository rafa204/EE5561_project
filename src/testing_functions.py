from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.criterion import *
from src.reference_net import CustomKLLoss


kl_loss_ref = CustomKLLoss()

def test_model(model, test_loader, net_type):
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
        for out_imgs, inp_imgs, mask in test_bar:
           
            
            if net_type == "VAE":
                central_index = params.num_slices//2
                central_slice = out_imgs[:,central_index,:,:].unsqueeze(1) #Get central slice for VAE output
                seg_out, vae_out, mu, logvar = model(inp_imgs)
                metrics[0] += soft_dice_coeff(seg_out, mask).mean()
                metrics[1] += MSE_loss(vae_out, central_slice)
                metrics[2] += kl_loss(mu, logvar)


            elif net_type == "ref":
                seg_y_pred, rec_y_pred, y_mid = model(inp_imgs)
                est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
                metrics[0] += soft_dice_coeff(seg_y_pred, mask).mean()
                metrics[1] += MSE_loss(rec_y_pred, out_imgs)
                metrics[2] += kl_loss_ref(est_mean, est_std)
        
    metrics = metrics / len(test_bar)
    
    print(f"--- Validation results --- DICE: {metrics[0]:3f}, MSE: {metrics[1]:3f}, KL {metrics[2]:3f}")
    
    return metrics

def bin_mask_2_multi(mask):
    mask_temp = torch.round(mask)
    num_classes = 3
    multi_mask = torch.zeros(mask.shape[1:3])
    for i in range(num_classes):
        multi_mask[mask_temp[i,:,:]==1] = i
    return multi_mask.squeeze()
    
    
def plot_examples(model, test_dataset, slices, save_path, net_type):
    
    model.eval()
    print("Plotting results" + "-"*60)
    
    with torch.no_grad():
        j = 0
        for i in slices:
            
            out_img, inp_img, mask = test_dataset[i]
            
            mask = mask.unsqueeze(0)
            out_img = out_img.unsqueeze(0)
            inp_img = inp_img.unsqueeze(0)
            

            
        
        if net_type == "VAE":
            
            seg_out, vae_out, mu, logvar = model(inp_img)
            seg_out_multi = bin_mask_2_multi(seg_out.squeeze())
            
            dice_coeff = soft_dice_coeff(seg_out, mask).mean()
            mse_loss = MSE_loss(vae_out, central_slice)
            
            central_idx = out_img.shape[1]//2
            central_slice = out_img[central_idx,:,:].unsqueeze(0).unsqueeze(0)
            mask_multi = bin_mask_2_multi(mask.squeeze())

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
            
        elif net_type == "UNET":
            
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
            
        elif net_type == "ref":
            seg_y_pred, rec_y_pred, y_mid = model(inp_img)
            
            print(seg_y_pred.shape)
            
            dice_coeff = soft_dice_coeff(seg_y_pred, mask).mean()
            mse_loss = MSE_loss(rec_y_pred, out_img)
            idx = seg_y_pred.shape[1]//2
            
            seg_pred_2d = bin_mask_2_multi(seg_y_pred[0,:,idx,:,:].squeeze())
            mask_2d = bin_mask_2_multi(mask[0,:,idx,:,:].squeeze())
            vae_out_2d = rec_y_pred[0,0,idx,:,:].squeeze()
            input_2d = out_img[0,0,idx,:,:].squeeze()
            
            fig, ax = plt.subplots(2,2,figsize = (10,5))
            plt.gray()
            ax[0,0].imshow(input_2d.cpu())
            ax[0,0].set_title("HR central slice")
            ax[0,1].imshow(mask_2d.cpu())
            ax[0,1].set_title("HR true mask")
            ax[1,0].imshow(seg_pred_2d.cpu())
            ax[1,0].set_title(f"Predicted mask | Dice = {dice_coeff:.3f}")
            ax[1,1].imshow(vae_out_2d.cpu())
            ax[1,1].set_title(f"VAE output | MSE = {mse_loss:.3f}")
            
            
        for ax in fig.get_axes():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='x', length=0)
            ax.tick_params(axis='y', length=0)
            
        fig.tight_layout()
        fig.show()
        fig.savefig(save_path / f"out_{j}.png")
        j += 1
        
    
    