# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:15:54 2023

@author: AmayaGS
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from image_preprocess_segmentation import SlideProcessor, SlideImage

# %%

MODES = ('WHOLE_SLIDE', 'PATCHES' )

DEFAULT_TARGET_DIR = r'C:\Users\Amaya\Documents\PhD\Data\IHC synovium segmentation'
MASK_TARGET_DIR = r'C:\Users\Amaya\Documents\PhD\Data\IHC synovium segmentation'

PATCH_SIZE = 224
STEP_SIZE = 224
slide_level = 2

stains = ["CD138", "CD68", "CD20"]


file = r"C:\Users\Amaya\Documents\PhD\Data\patient_labels.csv"
df = pd.read_csv(file, header=0)  

train_ids, test_ids = train_test_split(df['Patient ID'], test_size= 0.3, random_state= 1)
test_ids, val_ids = train_test_split(test_ids, test_size= 0.2, random_state= 1)

train_ids = list(train_ids)
test_ids = list(test_ids)
val_ids = list(val_ids)

# %%

for stain in stains:
    
    DEFAULT_SOURCE_DIR = r'C:\Users\Amaya\Documents\PhD\Data\R4RA slides\Nanozoomer scanned R4RA slides\NZ ' + stain + ' SCANNED IMAGES'
    DEFAULT_MASK_DIR = r'C:\Users\Amaya\Documents\PhD\Data\QuPath ' + stain + '\segmentation_masks'
    
    processor = SlideProcessor(source_dir = DEFAULT_SOURCE_DIR,
                               mask_source_dir = DEFAULT_MASK_DIR,
                               target_dir = DEFAULT_TARGET_DIR,
                               mask_target_dir = MASK_TARGET_DIR,
                               crop_factor = 0.01,
                               variance_factor = 0.75,
                               slide_level = slide_level,
                               train=train_ids,
                               test=test_ids,
                               val=val_ids)

    processor.run()
        
        
# %%

# mySlide2 = SlideImage(path=r"C:\Users\AmayaGS\Documents\PhD\Data\R4RA slides\R4RA slides\NZ CD20 SCANNED IMAGES\WHIP-R4RA-W999_CD20_PATH - 2021-02-08 12.44.58.ndpi", slide_level=3)
# threshold = mySlide2.calc_lum_threshold( mode = 'PEAK', criteria = 0.1)
# mySlide2.patchification()
# mySlide2.generate_mask(threshold)
# mySlide2.mask_patchification()
# mySlide2.calc_border_patches()
# mySlide2.crop_filter()    
# mySlide2.variance_filter(2.5)
# mySlide2.filter_patches()
# print(len(mySlide2.filtered_patches)) 