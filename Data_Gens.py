

        #### Import All libraies used for training  #####
        
import torch    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import PIL as Image

import matplotlib.pyplot as plt


  ### Data_Generators ########
  
NUM_WORKERS=0
PIN_MEMORY=True
batch_size= 24
height=  224
width= 224


train_transform = transforms.Compose([
        transforms.Resize((224, 224)),                            
        transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.1),
        transforms.ColorJitter(contrast=0.1), 
        transforms.ColorJitter(saturation=0.1),
        transforms.ColorJitter(hue=0.1)]),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      
    ])

test_transform = transforms.Compose([
        transforms.Resize((224, 224)),                            
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      
    ])


class Dataset_train(Dataset):
    
    def __init__(self, image_dir, mask_dir, train_transform):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = train_transform 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        image = Image.open(img_path)
        mask= cv2.imread(mask_path, 0)
         
        mask[np.where(mask!=0)]= 1
        mask = cv2.resize(mask, (height, width), interpolation = cv2.INTER_AREA) # We don't need to resize, as the GT is already 224 x 224, but we can leave as safety if you want. 
        
        mask= np.expand_dims(mask, axis=0)

        image_tensor = self.transform(image)

        return image_tensor, mask, self.images[index][:-4]
    

    
def Data_Loader_Train(image_dir, mask_dir, train_transform, batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_train(image_dir=image_dir, mask_dir=mask_dir, train_transform=train_transform)

    data_loader = DataLoader(test_ids, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    
    return data_loader


train_imgs='/data/scratch/acw676/keen/synthetic_2/test/img/'
train_masks='/data/scratch/acw676/keen/synthetic_2/test/gt1/'

train_loader = Data_Loader_Train(train_imgs, train_masks, train_transform, batch_size = 1)

a = iter(train_loader)
a1 =next(a)
plt.figure()
plt.imshow(a1[0][0,0,:,:])
plt.figure()
plt.imshow(a1[1][0,2,:,:])


