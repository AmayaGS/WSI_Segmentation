

#### Specify all the paths here #####

#### Specify all the paths here #####
test_imgs ='/data/scratch/wpw030/IHC_segmentation/Test/img/'   ## full-scale images 
test_masks ='/data/scratch/wpw030/IHC_segmentation/Test/GT/'  ## full-scale ground truths 
path_to_checkpoints = "/data/scratch/wpw030/IHC_segmentation/weights/UNet_512_1.pth.tar"
path_to_save_visual_results='/data/scratch/wpw030/IHC_segmentation/results_unet/'


import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import cv2
from empatches import EMPatches
emp = EMPatches()
import torch    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


import torch    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import cv2

import PIL
import PIL.Image as Image
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
from PIL import ImageFile

from torchvision import transforms
import matplotlib.pyplot as plt

import gc 
gc.enable()

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

  ### Data_Generators ########
  
NUM_WORKERS=0
PIN_MEMORY=True
height=224
width=224

test_transform = transforms.Compose([
        # transforms.Resize((224, 224)),                            
        transforms.ToTensor(),
        # transforms.Normalize((0.8946, 0.8659, 0.8638), (0.1050, 0.1188, 0.1180))      
    ])



class Dataset_Val(Dataset):
    
    def __init__(self, image_dir, mask_dir, transforms):
        self.image_dir = image_dir
        
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transforms
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index][:-4]+'.jpg')
        
        image = Image.open(img_path)
        image_tensor = self.transform(image)
        
        image_tensor = image_tensor.permute(1,2,0)
        image_tensor =np.array(image_tensor)

        img_patches, img_indices = emp.extract_patches(image_tensor, patchsize=224, overlap=0) 
        img_patches = np.array(img_patches)
        img_patches = img_patches.transpose(0,3,1,2)
        
        mask=cv2.imread(mask_path)
        mask[np.where(mask!=0)]=1
        mask=np.expand_dims(mask, axis=0)

        return img_patches, mask,img_indices, self.images[index][:-4]
    
def Data_Loader_Val( image_dir,mask_dir, test_transform, batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_Val(image_dir=image_dir, mask_dir=mask_dir, transforms=test_transform)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader

   ### Load the Data using Data generators and paths specified #####
   #######################################

test_loader = val_loader = Data_Loader_Val(test_imgs,test_masks,test_transform, batch_size = 1)
print(len(test_loader)) ### this shoud be = Total_images/ batch size

def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
    
def blend(image1,gt,pre, ratio=0.5):
    
    assert 0 < ratio <= 1, "'cut' must be in 0 to 1"

    alpha = ratio
    beta = 1 - alpha
    theta=beta-0.1

    #coloring yellow.
    gt *= [0.2,0.7, 0] ### Green Color
    pre*=[1,0,0]   ## Red Color
    image = image1 * alpha + gt * beta+ pre * theta
    return image
    

test_transform1 = transforms.Compose([
        # transforms.Resize((224, 224)),                            
        # transforms.ToTensor(),
        transforms.Normalize((0.8946, 0.8659, 0.8638), (0.1050, 0.1188, 0.1180))      
    ])


def check_dice_score(test_loader, model, device=DEVICE):
	
    loop = tqdm(test_loader)
    model.eval()
    Avg_Dice_Score = 0
	
    with torch.no_grad():
        for batch_idx, (img_patches,t1,mask_indices,label) in enumerate(loop):
            
            img_patches = img_patches.to(device=DEVICE,dtype=torch.float)[0,:,:,:,:]
            t1 = t1.to(device=DEVICE,dtype=torch.float)
            
            pred1=[]
            for i in range(img_patches.shape[0]):
				
                single_img = img_patches[i,:,:,:]
                
                single_img = test_transform1(single_img)
                
                single_img=torch.unsqueeze(single_img, axis=0)
                p1=model(single_img)
                              
                p1 = (p1 > 0.5) * 1
                
                p1=p1.cuda().cpu()
                p1 = np.stack((p1,)*3, axis=-1)
                pred1.append(p1)

            merged_pre = emp.merge_patches(pred1, mask_indices)
                    
            t1_cpu = t1.to('cpu')    
            t1_cpu= t1_cpu[0,:,:,:].numpy()
            
            dice_score = (2 * (merged_pre * t1_cpu).sum()) / (
                (merged_pre + t1_cpu).sum() + 1e-8
            )
            
            Avg_Dice_Score +=dice_score
            #dice_score=round(dice_score,3)
            
            print('Dice Score for image',label[0], 'is :',dice_score)
                        
    
    print(f"Average Dice Score is  : {Avg_Dice_Score/len(test_loader)}")

from Models import UNet_512
Model = UNet_512()    
LEARNING_RATE=0.0
optimizer = optim.Adam(Model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
      
def eval_():
    Model.to(device=DEVICE,dtype=torch.float)
    checkpoint = torch.load(path_to_checkpoints,map_location=DEVICE)
    Model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    check_dice_score(test_loader, Model, device=DEVICE)

    
if __name__ == "__main__":
    eval_()
