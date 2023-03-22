height=224
width=224

        #### Import All libraies used for training  #####
import torch    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import cv2
import torchio as tio
import matplotlib.pyplot as plt
  ### Data_Generators ########
  
NUM_WORKERS=0
PIN_MEMORY=True

class Dataset_train(Dataset):
    def __init__(self, image_dir, mask_dir ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index][:-4]+'.jpg')
        
        image=cv2.imread(img_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        image = image/255
        image = image.transpose(2,1,0)
        #image = np.expand_dims(image, axis=0)

        
        mask=cv2.imread(mask_path,0)
        mask[np.where(mask!=0)]=1
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)
        
        mask=np.expand_dims(mask, axis=0)
        #mask=np.expand_dims(mask, axis=0)

        # d = {}
        # d['Image'] = tio.Image(tensor = image, type=tio.INTENSITY)
        # d['Mask'] = tio.Image(tensor = mask, type=tio.LABEL)
        # sample = tio.Subject(d)
        
        # if self.transformations is not None:
        #     transformed_tensor = self.transformations(sample)
        #     image = transformed_tensor['Image'].data
        #     mask = transformed_tensor['Mask'].data
        

        return image,mask,self.images[index][:-4]
    
def Data_Loader_Train( image_dir,mask_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_train( image_dir=image_dir, mask_dir=mask_dir)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

class Dataset_val(Dataset):
    def __init__(self, image_dir, mask_dir ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index][:-4]+'.jpg')
        
        image=cv2.imread(img_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        image = image/255
        image = image.transpose(2,1,0)
        #image = np.expand_dims(image, axis=0)

        
        mask=cv2.imread(mask_path,0)
        mask[np.where(mask!=0)]=1
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)
        
        mask=np.expand_dims(mask, axis=0)
        #mask=np.expand_dims(mask, axis=0)

        # d = {}
        # d['Image'] = tio.Image(tensor = image, type=tio.INTENSITY)
        # d['Mask'] = tio.Image(tensor = mask, type=tio.LABEL)
        # sample = tio.Subject(d)
        
        # if self.transformations is not None:
        #     transformed_tensor = self.transformations(sample)
        #     image = transformed_tensor['Image'].data
        #     mask = transformed_tensor['Mask'].data
        

        return image,mask,self.images[index][:-4]
    
def Data_Loader_Val( image_dir,mask_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_val( image_dir=image_dir, mask_dir=mask_dir)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


train_imgs = r'C:\My_Data\Amaya\Train\img'
train_masks = r'C:\My_Data\Amaya\Train\GT'
train_loader = Data_Loader_Val(train_imgs,train_masks,batch_size = 2)
train_loader = Data_Loader_Train(train_imgs,train_masks,batch_size = 2)

a = iter(train_loader)
a1 =next(a)
img = a1[0][0,:].numpy()
img = img.transpose(2,1,0)
gt = a1[1][0,0,:].numpy()
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(gt)
