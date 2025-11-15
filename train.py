import sys
from tqdm import tqdm
from src.data_preparation import *
from src.network import *
from src.criterion import *
from src.testing_functions import *
# from src.reference_network import *
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import pickle

#=========== SETUP PARAMETERS ===============

label = "test3" #This is the name of the results folder
       
params = Training_Parameters() #Initialize training parameters, you can change them in src/data_preparation.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = '../BRATS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'

#Directory for output results
results_path = Path('training_results')/label
results_path.mkdir(parents=True, exist_ok=True)

validation_metrics = np.zeros((params.num_epochs,3))

#=========== SETUP DATASETS AND DATA LOADERS ===============

dataset = BRATS_dataset(data_path, device, params)

#Create training and validation datasets
train_size = int(params.train_ratio * len(dataset))
val_size = len(dataset) - train_size

# For reproducibility, set a random seed
g = torch.Generator().manual_seed(42) 
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=g
)

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)


#=========== SETUP MODEL AND OPTIMIZER ===============

#With VAE branch
if params.VAE: 
    model = VAE_UNET(params.num_slices, input_dim=dataset.input_dim, HR_dim=dataset.output_dim)
    #model = NvNet(3, [240,240], 3, "relu", "group_normalization", "True", mode='bilinear')

else: #Without VAE branch (only UNET)
    model = UNET(params.num_slices)
    
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = params.learning_rate, weight_decay = 1e-5)

# LR Scheduler
lr_lambda = lambda epoch: (1 - epoch / params.num_epochs) ** 0.9
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
criterion = combined_loss #For the VAE option, not UNET


#=========== TRAINING LOOP ===============
print("Starting training "+ label)

for epoch in range(params.num_epochs):
    
    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params.num_epochs} [Training]")

    for out_imgs, inp_imgs, mask in train_bar:
        
 
        central_index = params.num_slices//2
        central_slice = out_imgs[:,central_index,:,:].unsqueeze(1) #Get central slice for VAE output

        optimizer.zero_grad()
        if params.VAE:
            seg_out, vae_out, mu, logvar = model(inp_imgs)
            loss = criterion(seg_out, mask, vae_out, central_slice, mu, logvar)
        else:
            seg_out = model(inp_imgs)
            loss = dice_loss(seg_out, mask)
        
        
        loss.backward()
        optimizer.step()
        
        train_bar.set_postfix(loss=loss.item())
    
    
    #---Validation    
    if(params.validation):
        validation_metrics[epoch,:] = test_model(model, val_loader, params.VAE)
        np.save(results_path / "training_metrics.npy", validation_metrics)
    
     #---Logging 
    if(params.save_model_each_epoch):
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, results_path / "checkpoint.pth.tar")
        with open(results_path /'params.pkl', 'wb') as f:
            pickle.dump(params, f)

  
    scheduler.step() #Adjust learning rate
