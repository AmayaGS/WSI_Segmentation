# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:53:04 2024

@author: AmayaGS
"""# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:17:53 2023

@author: AmayaGS
"""


#### Specify all the paths here #####
# test_imgs ='/data/scratch/wpw030/IHC_segmentation/Test/img/'   ## full-scale images
# test_masks ='/data/scratch/wpw030/IHC_segmentation/Test/GT/'  ## full-scale ground truths
# path_to_checkpoints = "/data/scratch/wpw030/IHC_segmentation/weights/UNet_512_1.pth.tar"
# path_to_save_visual_results='/data/scratch/wpw030/IHC_segmentation/results/'

test_imgs = r"C:\Users\Amaya\Desktop\test_img/"

#PATH = r"C:\Users\Amaya\Documents\PhD\Data\IHC synovium segmentation"
#test_imgs = PATH + "\Test\img/"   ## full-scale images
#test_masks =  PATH + "\Test\GT/"  ## full-scale ground truths
path_to_checkpoints = r"C:\Users\Amaya\Documents\PhD\IHC-segmentation\weights\UNet_512_1.pth.tar"
path_to_save_visual_results= r"C:\Users\Amaya\Desktop\test\results/"
path_to_save_patches = r"C:\Users\Amaya\Desktop\test\results/patches/"


import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

#import PIL
import PIL.Image as Image
import openslide as osi
#from patchify import patchify
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
#import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

#import albumentations as A
#from albumentations.pytorch import ToTensorV2
#import segmentation_models_pytorch as smp

#from empatches import EMPatches, BatchPatching
from empatches import EMPatches
emp = EMPatches()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import gc
gc.enable()

# Variables
NUM_WORKERS = 0
PIN_MEMORY = True
patchsize = 224
overlap = 0
shuffle = True
keep_patches = True
coverage = 0.2
slide_batch = 1
patch_batch = 10

### Data_Generators ########

class Dataset_Val(Dataset):

    def __init__(self, image_dir, transform, slide_level=2):

        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
        self.slide_level = slide_level

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.images[index])

        slide = osi.OpenSlide(img_path)
        #properties = slide.properties
        #adjusted_level = int(slide_level + np.log2(int(properties['openslide.objective-power'])/40))
        slide_adjusted_level_dims = slide.level_dimensions[self.slide_level]
        image = np.array(slide.read_region((0, 0), self.slide_level, slide_adjusted_level_dims).convert('RGB'))
        #image = cv2.imread(img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = Image.open(img_path)
        # image_tensor = self.transform(image)

        # image_tensor = image_tensor.permute(1,2,0)
        # image_tensor = np.array(image_tensor)

        img_patches, img_indices = emp.extract_patches(image, patchsize=patchsize, overlap=overlap)
        #patches_img = patchify(cropped_image, (patch_size, patch_size, 3), step=step)
        #img_patches = np.array(img_patches)
        #img_patches = img_patches.transpose(0, 3, 1, 2)

        if self.transform:
            img_patches = [self.transform(img) for img in img_patches]
            #img_patches = torch.stack(img_patches)
            #img_patches = np.array(img_patches)
        #img_patches = torch.tensor(img_patches)

        return img_patches, img_indices, self.images[index][:-4]

def slide_dataloader(image_dir, transform, slide_batch=1, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY):

    dataset = Dataset_Val(image_dir=image_dir, transform=transform)
    slide_loader = DataLoader(dataset, batch_size=slide_batch, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)

    return slide_loader


   ### Load the Data using Data generators and paths specified #####
   #######################################

mean = (0.8946, 0.8659, 0.8638)
std = (0.1050, 0.1188, 0.1180)

test_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

# a_transform = A.Compose([
#     A.Normalize((0.8946, 0.8659, 0.8638), (0.1050, 0.1188, 0.1180)),
#     ToTensorV2()
# ])

test_loader = slide_dataloader(test_imgs, test_transform, slide_batch = slide_batch)
print(len(test_loader)) ### this shoud be = Total_images/ batch size

def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def blend(image1, gt, pre, ratio=0.5):

    assert 0 < ratio <= 1, "'cut' must be in 0 to 1"

    alpha = ratio
    beta = 1 - alpha
    theta=beta-0.1

    #coloring yellow.
    gt *= [0.2,0.7, 0] ### Green color
    pre*=[1,0,0]   ## Red color
    image = image1 * alpha + gt * beta + pre * theta
    return image

def dice_coefficient(y_true, y_pred):

    try:
        intersection = np.sum(y_true * y_pred)
        return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)
    except TypeError:
        return 0.

def batch_generator(items, batch_size):
    count = 1
    chunk = []

    for item in items:
        if count % batch_size:
            chunk.append(item)
        else:
            chunk.append(item)
            yield chunk
            chunk = []
        count += 1

    if len(chunk):
        yield chunk

def check_dice_score(test_loader, model, batch_size=patch_batch, mean=mean, std=std, device=DEVICE):

    loop = tqdm(test_loader)
    #Avg_Dice_Score = 0

    model.eval()
    with torch.no_grad():

        f = open(path_to_save_visual_results + "extracted_patches.csv", "w")
        f.write("Patient_ID,Filename,Patch_name,Patch_coordinates,File_location\n")

        for batch_idx, (img_patches, img_indices, label) in enumerate(loop):

            name = label[0]
            patient_ID = label[0].split("_")[0]

            #img_patches = np.squeeze(img_patches)
            num_patches = len(img_patches)

            print(f"Processing WSI: {name}, with {num_patches} patches")

            pred1 = []

            for i, batch in enumerate(batch_generator(img_patches, batch_size)):
                batch = np.squeeze(torch.stack(batch))
                batch = batch.to(device=DEVICE, dtype=torch.float)
                #print(batch.shape)
            # for i in range(0, img_patches.shape[0], batch_size):

            #     single_img = img_patches[i:i+batch_size,:,:,:]
            #     #single_img = img_patches[i:i+batch_size]
            #     single_img = single_img.to(device=DEVICE, dtype=torch.float)

                p1 = model(batch)
                p1 = (p1 > 0.5) * 1
                p1= p1.detach().cpu()
                pred_patch_array = np.squeeze(p1)

                for b in pred_patch_array:
                    pred1.append(b)

                if keep_patches:

                    for patch in range(len(pred_patch_array)):
                        #print(patch, len(pred_patch_array))

                        white_pixels = np.count_nonzero(pred_patch_array[patch])

                        if (white_pixels / len(pred_patch_array[patch])**2) > coverage:

                            patch_image = batch[patch].detach().cpu().numpy().transpose(1, 2, 0)
                            patch_image[:, :, 0] = patch_image[:, :, 0] * std[0] + mean[0]
                            patch_image[:, :, 1] = patch_image[:, :, 1] * std[1] + mean[1]
                            patch_image[:, :, 2] = patch_image[:, :, 2] * std[2] + mean[2]

                            patch_loc_array = np.array(torch.cat(img_indices[i*batch_size + patch]))
                            patch_loc_str = f"_x={patch_loc_array[0]}_x+1={patch_loc_array[1]}_y={patch_loc_array[2]}_y+1={patch_loc_array[3]}"
                            patch_name = name + patch_loc_str + ".png"
                            folder_location = os.path.join(path_to_save_patches, patient_ID)
                            os.makedirs(folder_location, exist_ok=True)
                            file_location = folder_location + "/" + patch_name
                            #print(patch_image, np.min(patch_image), np.max(patch_image))
                            plt.imsave(file_location, patch_image)

                            f.write("{},{},{},{},{}\n".format(patient_ID,name,patch_name,patch_loc_array,file_location))

                del p1, batch, pred_patch_array
                gc.collect()

            merged_pre = emp.merge_patches(pred1, img_indices)

            #dice_score = dice_coefficient(mask, merged_pre)

            #Avg_Dice_Score += dice_score

            #print('Dice Score for image', name, 'is :', dice_score)

            #merged_pre[np.where(merged_pre[:]!=0)]=1

            plt.imsave(os.path.join(path_to_save_visual_results, name +".png"), merged_pre)

            del merged_pre, pred1
            gc.collect()

        f.close


from Models import UNet_512
Model = UNet_512()


def eval_():
    Model.to(device=DEVICE,dtype=torch.float)
    checkpoint = torch.load(path_to_checkpoints, map_location=DEVICE)
    Model.load_state_dict(checkpoint['state_dict'], strict=True)

    check_dice_score(test_loader, Model, device=DEVICE)


# %%

if __name__ == "__main__":
    eval_()