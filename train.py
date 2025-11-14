import sys
from tqdm import tqdm
from src.data_preparation import *
from src.network import *
from src.criterion import *
from src.testing_functions import *
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path

#=========== SETUP PARAMETERS ===============

label = "test2"
num_slices = 3
downsamp_type = 'bilinear'
ds_ratio = 2

num_epochs = 1000
learning_rate = 1e-5
batch_size = 1
validation = True
save_model_each_epoch = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = '../BRATS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
results_path = Path('training_results')/label
results_path.mkdir(parents=True, exist_ok=True)

validation_metrics = np.zeros((num_epochs,3))

#=========== SETUP DATASETS AND DATA LOADERS ===============

dataset = BRATS_dataset(data_path, device, num_slices = num_slices, ds_ratio = ds_ratio, downsamp_type = downsamp_type, slices_per_volume = 1, num_volumes = 2)

# Determine Split Ratios
train_ratio = 0.5
val_ratio = 0.5

train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size # Ensure sum equals total_samples

# For reproducibility, set a random seed
g = torch.Generator().manual_seed(42) 
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=g
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = train_loader

val_loader = DataLoader(val_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)

#=========== SETUP MODEL AND OPTIMIZER ===============

model = VAE_UNET(num_slices, input_dim=dataset.input_dim, HR_dim=dataset.output_dim)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)

# LR Scheduler
lr_lambda = lambda epoch: (1 - epoch / num_epochs) ** 0.9
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
criterion = combined_loss


#=========== TRAINING LOOP ===============
print("Starting training")

for epoch in range(num_epochs):
    
    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")

    for img_list, ds_img_list, mask in train_bar:
        
        if torch.all(img_list[0] < 1e-8): #Ignore slices with no brain
            continue
            
        central_slice = img_list[0][:,1,:,:].unsqueeze(1)

        optimizer.zero_grad()
        seg_out, vae_out, mu, logvar = model(ds_img_list[0])
        loss = criterion(seg_out, mask, vae_out, central_slice, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        train_bar.set_postfix(loss=loss.item())
    
    
    #---Validation    
    if(validation):
        validation_metrics[epoch,:] = test_model(model, val_loader)
        np.save(results_path / "training_metrics.npy", validation_metrics)
    
    #---Logging 
    if(save_model_each_epoch):
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'num_slices' = num_slices
        'downsamp_type' = downsamp_type
        'downsamp_type' = ds_ratio
        'num_epochs' = num_epochs
        'learning_rate' = learning_rate
        'batch_size' = batch_size
        'validation' = True
        }
        torch.save(checkpoint, results_path / "checkpoint.pth.tar")

  
    scheduler.step() #Adjust learning rate
