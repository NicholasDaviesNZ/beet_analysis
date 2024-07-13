"""
Having done the pretty poor manual masking on ~20 plots, this will not work well enough to be used for anything, but we will run though it anyway. Just to do a bit of few-shot transfer learning in pytorch 
This wont get used for anything other than being displayed on github
More cleaner labels would definitly be required
Would be interesting to try with some ratios of color bands - RG ratios etc.

Transfer learning to try and automate image segmentation of beet plants and weeds from a herbicide trial to quantify overall herbicide efficacy on feild pansy (vertually the only weed present in the trial)
and a breif look at safety on beet. Note this is in conjunction with field measurments and will nto be used for research. 
"""

import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToPILImage
torch.cuda.empty_cache()

path_to_images_with_masks = "/workspaces/beet_analysis/images_with_masks/"
path_to_masks = "/workspaces/beet_analysis/images_masks/"
path_to_all_images = "/workspaces/beet_analysis/plot_images_resized/"
batch_size = 32
num_epochs = 500
learning_rate = 0.00001

class BeetDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = os.listdir(images_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        # Load images
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.pad_image(image)
        mask = self.pad_image(mask)
        
        # Apply transformations if defined
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, img_name
    
    def pad_image(self, image, target_size = (128,192)):
        width, height = image.size
        target_width, target_height = target_size

        pad_width = target_width - width
        pad_height = target_height - height
        padding = (0, 0, pad_width, pad_height)  

        padded_image = transforms.functional.pad(image, padding, fill=0)
       
        return padded_image
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(contrast=0.5),
    transforms.RandomRotation(30),
    transforms.CenterCrop(480),
])

# built train and test datasets
dataset = BeetDataset(path_to_images_with_masks, path_to_masks, transform=transform)
train_size = int(0.6*len(dataset))
val_size = int(0.2*len(dataset))
test_size = len(dataset) - val_size - train_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# get a pretrained unet from efficientnet for rgb images, and 3 seg classes
model = smp.Unet(
    encoder_name="timm-efficientnet-b0", 
    in_channels=3,                  
    classes=3,                      
)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
        
# training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets, _ in train_loader:
        inputs = inputs.to(device)
        targets = targets.squeeze(1).long().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

    if (epoch + 1) % 1 == 0: 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(device)
                targets = targets.squeeze(1).long().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}")


output_dir = '/workspaces/beet_analysis/predicted_masks/'

# test loop
model.cpu()
model.eval()
test_loss = 0.0

with torch.no_grad():
    for i, (inputs, targets, image_names) in enumerate(test_loader):
        targets = targets.squeeze(1).long()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
        
        # Convert outputs to mask
        _, predicted = torch.max(outputs, 1)
        
        # Convert tensor to uint8
        predicted = predicted.cpu().byte()
        
        to_pil = ToPILImage()
        for j in range(predicted.size(0)):
            image_name = image_names[j]
            print(image_name)
            predicted_mask = to_pil(predicted[j]*100)
            output_path = os.path.join(output_dir, f"{image_name}")
            predicted_mask.save(output_path)
            print(f"Saved predicted mask to {output_path}")
        
        print(f"Saved predicted mask to {output_path}")

test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")