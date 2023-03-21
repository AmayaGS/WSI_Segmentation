# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 19:15:54 2023

@author: AmayaGS
"""

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))


import cv2
import openslide as osi
import tifffile as tiff

from patchify import patchify

#from skimage import color
#from skimage import morphology
#from skimage.io import imread


# %%

class SlideImage():
    
    def __init__(self, path, mask_path, slide_level=3, color_channels = "RGB", crop_factor = 1 ):
        
        self.slide = osi.OpenSlide(path)
        #self.image = self.slide.read_region((0, 0), slide_level, self.slide.level_dimensions[slide_level]).convert(color_channels)
        self.path = path
        self.img_ID = self.path.split('\\')[-1].split('_')[0].split('.')[0]
        self.properties = self.slide.properties
        self.adjusted_level = int(slide_level + np.log2(int(self.properties['openslide.objective-power'])/40))
        self.image_array = np.array(self.slide.read_region((0, 0),
                                                           self.adjusted_level,
                                                           self.slide.level_dimensions[self.adjusted_level]).convert(color_channels))
        self.n_channels = self.image_array.shape[-1]
        
        np_mask = cv2.imread(mask_path, 0)
        self.mask = cv2.resize(np_mask, self.slide.level_dimensions[self.adjusted_level])
        
        self.lum_array = None
        self.grayscale = None
        self.patches = None
        #self.mask = None
        self.border_patches = []
        self.bad_mask_patches = []
        
        self.mask_patches = None
        
        
        self.filtered_patches = []
        self.filtered_mask_patches = []
        self.white_patches = []
        self.bad_patches = []
        self.cropped_patches = []
        self.crop_factor = crop_factor
        # if use_mask:
        #     self.patchification()
        #     self.mask = self.generate_mask(self.calc_lum_threshold())

    # def print_properties(self):
        
    #     print(self.slide.level_count)
    #     print(self.slide.dimensions)
    #     print(self.slide.level_count)
    #     print(self.slide.level_downsamples)
    #     print(self.slide.level_dimensions)

    def patchification(self, width=PATCH_SIZE, height=PATCH_SIZE, step=STEP_SIZE):
        self.patches = patchify(self.image_array, (width, height, self.n_channels ), step=step)
        
    def mask_patchification(self, width=PATCH_SIZE, height=PATCH_SIZE, step=STEP_SIZE):
        self.mask_patches = patchify(self.mask, (width, height), step=STEP_SIZE)
        
    # def generate_mask(self, lum_threshold=0.9):
    #     lum = color.rgb2gray(self.image_array)
    #     mask = morphology.remove_small_holes( morphology.remove_small_objects( lum < lum_threshold, 500), 500)
    #     mask_bool = morphology.opening(mask, morphology.disk(3))
    #     self.mask = mask_bool.astype('uint8') * 255
    
    def calc_lum_array(self):
        if self.patches is None:
            self.patchification()
        self.grayscale = np.dot(self.patches, 1/255 * np.array([0.2125, 0.7154, 0.0721]))
        self.lum_array = self.grayscale.sum(axis=(2,3,4)) * (1 / PATCH_SIZE ** 2 )      
           
    def calc_lum_threshold(self, mode = 'MEAN', criteria = 0.1 ):
        
        if self.lum_array is None or self.grayscale is None:
            self.calc_lum_array()
                            
        if mode == 'MEAN':
            return self.lum_array.mean()
        elif mode == 'PEAK':
            histogram, bin_edges = np.histogram(self.lum_array, bins=2000, range=(0, 1))                        
            max_count = max(histogram) 
            max_count_admitted = max_count * criteria
            index_max = np.argmax(histogram)
            i = index_max
            while histogram[i] > max_count_admitted:
                i -= 1
            return bin_edges[i]
        else:
            raise RuntimeError('Unknown threshold calc mode')
            
    # def mask_plot(self):    
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
    #     ax.set_title("mask")
    #     plt.grid(None) 
    #     ax.imshow(self.mask)

    # def image_plot(self):    
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
        
    #     ax.imshow(self.image_array)
    #     plt.grid(None) 
    #     ax.set_title("slide")
    

    def filter_patches(self, method = 'MASK', params = {'coverage': 0.9}):
        
        self.filtered_patches = [] 
        self.bad_patches = [] 
        self.filtered_mask_patches = []
        self.bad_mask_patches = []
        
        for i in range(self.patches.shape[0]):
                for j in range(self.patches.shape[1]):
                    
                    single_patch_mask = self.mask_patches[i,j,:,:]
                    single_patch_img = self.patches[i,j,:,:]
                    
                    white_pixels = np.count_nonzero(single_patch_mask)    
                    
                    #if white_pixels != 0 or (i,j) in (self.border_patches + self.white_patches):
                    if white_pixels != 0:
                        self.filtered_patches.append((single_patch_img, i, j))
                        self.filtered_mask_patches.append((single_patch_mask, i, j))
                                             
                    if (i,j) in (self.border_patches + self.white_patches):
                        
                        self.bad_patches.append((single_patch_img, i, j))
                        self.bad_mask_patches.append((single_patch_mask, i, j))
                    
                    # if (i,j) in (self.white_patches):
                    #     self.bad_patches.append((single_patch_img, i, j))
                    #     self.bad_mask_patches.append((single_patch_mask, i, j))
                        
                    # if white_pixels/ single_patch_mask.size > params['coverage'] and (i, j) not in (self.border_patches + self.white_patches + self.cropped_patches):
                        
    def calc_border_patches(self, max_zscore = 10, ):
        self.border_patches = []
        if self.lum_array is None or self.grayscale is None:
            self.calc_lum_array()
        global_std = self.lum_array.std()
        global_mean = self.lum_array.mean()
        border_patches = []
        for i in range(len(self.lum_array)):
            for j in range(len(self.lum_array[i])):
                single_patch_img = self.patches[i,j,:,:]
                single_patch_gray = 1/255 * np.dot(single_patch_img, np.array([0.2125, 0.7154, 0.0721]))[0]
                min_rows_avg = min(single_patch_gray.mean(axis = 1))
                min_cols_avg = min(single_patch_gray.mean(axis = 0))
                min_lum = min(min_rows_avg,min_cols_avg)
                if min_lum < global_mean - max_zscore * global_std:
                    border_patches.append((i,j))
                    #print(i,j)
        self.border_patches = border_patches            


    def variance_filter(self, factor = 1., cropped = True ):
        if self.patches is None:
            self.patchification()
        
        self.white_patches = []
        
        xsize = self.patches.shape[0]
        ysize = self.patches.shape[1]
        
        if cropped:

            max_x = int(xsize * (1 + self.crop_factor) / 2)
            max_y = int(ysize * (1 + self.crop_factor) / 2)        
            min_x = int(xsize * (1 - self.crop_factor) / 2)
            min_y = int(ysize * (1 - self.crop_factor) / 2)      
            
            c_stds = [np.std(self.patches[i][j]) for i in range(min_x,max_x) for j in range(min_y, max_y)]

        else:
            c_stds = [np.std(self.patches[i][j]) for i in range(0,xsize) for j in range(0, ysize)]
        
        cutoff = factor * np.percentile(c_stds, 50)
        
        for i in range(self.patches.shape[0]):
                for j in range(self.patches.shape[1]):
                    if np.std(self.patches[i,j,:,:]) < cutoff:
                        self.white_patches.append((i,j))

    def crop_filter(self, factor = 0.7 ):
        self.check_patches()
        
        xsize = self.patches.shape[0]
        ysize = self.patches.shape[1]
        
        max_x = int(xsize * (1 + factor) / 2)
        max_y = int(ysize * (1 + factor) / 2)
        
        min_x = int(xsize * (1 - factor) / 2)
        min_y = int(ysize * (1 - factor) / 2)    
        
        self.cropped_patches = [(i,j) for i in range(0, min_x) for j in range(ysize)]
        self.cropped_patches += [(i,j) for i in range(max_x, xsize) for j in range(ysize)]
        self.cropped_patches += [(i,j) for i in range(xsize) for j in range(0,min_y)]
        self.cropped_patches += [(i,j) for i in range(xsize) for j in range(max_y, ysize)]
                               
    def write_filtered_patches(self, path):
        
        for i, patch in enumerate(self.filtered_patches):
                filename = path + '\\%s_%d_%d.png' % (id, patch[-2], patch[-1])  
                tiff.imwrite(filename,patch[0])
        

    def check_patches(self):
        if self.patches is None:
            self.patchification()
        

class SlideProcessor():
    
    def __init__(self, source_dir = DEFAULT_SOURCE_DIR,
                 mask_source_dir = DEFAULT_MASK_DIR,
                 target_dir = DEFAULT_TARGET_DIR,
                 mask_target_dir = MASK_TARGET_DIR,
                 patch_size = PATCH_SIZE,
                 step = STEP_SIZE,
                 train = train_ids,
                 test = test_ids,
                 val = val_ids,
                 index_file = None,
                 mode = 'PATCHES',
                 slide_level=3,
                 filter_threshold = 'PEAK',
                 crop_factor = 0.75,
                 variance_factor = 2.5,
                 use_mask = True):
        
        if mode not in MODES:
            raise ValueError('mode must be in', MODES)
            
        self.source_dir = source_dir
        self.mask_source_dir = mask_source_dir
        self.target_dir = target_dir
        self.mask_target_dir = mask_target_dir
        self.patch_size = patch_size
        self.step = step
        self.train_ids = train
        self.test_ids = test
        self.val_ids = val
        self.mode = mode
        self.index_file = index_file
        self.file_list = self._file_list()
        self.mask_list = self._mask_list()
        self.use_mask = use_mask
        self.filter_threshold = filter_threshold
        self.slide_level = slide_level
        self.variance_factor = variance_factor
        self.crop_factor = crop_factor
        
    def _file_list(self):
        return [f.path for f in os.scandir(self.source_dir)]  
    
    def _mask_list(self):
        return [f.path for f in os.scandir(self.mask_source_dir)]

    def run(self):
        
        files = sorted(self._file_list(), reverse = True)
        masks = sorted(self._mask_list(), reverse = True)
        
        #img_ID_counts = {}
            
        print('Processing %d files' % len(files))
        
        for file in files:
            
            file_name = file.split("\\")[-1][0:-5]
            patient_id = file.split("\\")[-1].split("_")[0]
            
            for mask in masks:
                
                mask_name = mask.split("\\")[-1][0:-4]
                
                if mask_name == file_name:
            
                    if patient_id in self.train_ids:

                        try:
                            
                            slideImg = SlideImage(path = file, mask_path= mask, slide_level = self.slide_level, color_channels = "RGB" )
                            
                            file_path = self.target_dir + '\\Train\\img'
                            mask_path = self.mask_target_dir + '\\Train\\GT'
                            #os.makedirs(path, exist_ok = True)
                            
                            #id = slideImg.img_ID
                            
                            # if id not in img_ID_counts:
                            #     img_ID_counts[id] = 0
                            # else:
                            #     img_ID_counts[id] += 1
                                
                            print('Processing %s' % file_name)  
                            
                            if self.mode == 'PATCHES':
                                
                                slideImg.patchification(width=self.patch_size, height=self.patch_size, step=self.step)
                                
                            if self.use_mask:
                                
                                # threshold = slideImg.calc_lum_threshold( mode = self.filter_threshold, criteria = 0.1 )
                                # slideImg.generate_mask(threshold)
                                slideImg.mask_patchification(width=self.patch_size, height=self.patch_size, step=self.step)       
                            
                            slideImg.calc_border_patches()
                            #slideImg.crop_filter(self.crop_factor)
                            slideImg.variance_filter(self.variance_factor)
                            #slideImg.calc_border_patches() ## hace falta dos veces??
                            
                            slideImg.filter_patches()
                
                            for i, (patch, mask) in enumerate(zip(slideImg.filtered_patches, slideImg.filtered_mask_patches)):
                                
                                filename = file_path + '\\%s_%d_%d.png' % (file_name, patch[-2], patch[-1])  
                                maskname = mask_path + '\\%s_%d_%d.jpg' % (file_name, mask[-2], mask[-1])  
                                
                                result_BGR = cv2.cvtColor(patch[0].squeeze(), cv2.COLOR_RGB2BGR)
                                
                                cv2.imwrite(filename, result_BGR)
                                cv2.imwrite(maskname, mask[0].squeeze())
                            
                            for i, (bad_patch, bad_mask) in enumerate(zip(slideImg.bad_patches, slideImg.bad_mask_patches)):
                                
                                filename = file_path + '\\%s_%d_%d.png' % (file_name, bad_patch[-2], bad_patch[-1])  
                                maskname = mask_path + '\\%s_%d_%d.jpg' % (file_name, bad_mask[-2], bad_mask[-1]) 
                                
                                result_BGR = cv2.cvtColor(bad_patch[0].squeeze(), cv2.COLOR_RGB2BGR)
                                                            
                                cv2.imwrite(filename, result_BGR)
                                cv2.imwrite(maskname, bad_mask[0].squeeze())
                                
                            print('Done')
                                
                        except Exception as ex:
                            print('Failed to process %s with error: %s' % (file, ex) )
                    
                        else:
                            continue
                
                
                    elif patient_id in self.test_ids:
                                
                        try:
                            
                            slideImg = SlideImage(path = file, mask_path= mask, slide_level = self.slide_level, color_channels = "RGB" )
                            
                            file_path = self.target_dir + '\\Test\\img'
                            mask_path = self.mask_target_dir + '\\Test\\GT'
                            #os.makedirs(path, exist_ok = True)
                            
                            #id = slideImg.img_ID
                            
                            # if id not in img_ID_counts:
                            #     img_ID_counts[id] = 0
                            # else:
                            #     img_ID_counts[id] += 1
                                
                            print('Processing %s' % file_name)  
                            
                            if self.mode == 'PATCHES':
                                
                                slideImg.patchification(width=self.patch_size, height=self.patch_size, step=self.step)
                                
                            if self.use_mask:
                                
                                # threshold = slideImg.calc_lum_threshold( mode = self.filter_threshold, criteria = 0.1 )
                                # slideImg.generate_mask(threshold)
                                slideImg.mask_patchification(width=self.patch_size, height=self.patch_size, step=self.step)       
                            
                            slideImg.calc_border_patches()
                            #slideImg.crop_filter(self.crop_factor)
                            slideImg.variance_filter(self.variance_factor)
                            #slideImg.calc_border_patches() ## hace falta dos veces??
                            
                            slideImg.filter_patches()
                
                            for i, (patch, mask) in enumerate(zip(slideImg.filtered_patches, slideImg.filtered_mask_patches)):
                                
                                filename = file_path + '\\%s_%d_%d.png' % (file_name, patch[-2], patch[-1])  
                                maskname = mask_path + '\\%s_%d_%d.jpg' % (file_name, mask[-2], mask[-1])  
                                
                                result_BGR = cv2.cvtColor(patch[0].squeeze(), cv2.COLOR_RGB2BGR)
                                
                                cv2.imwrite(filename, result_BGR)
                                cv2.imwrite(maskname, mask[0].squeeze())
                            
                            for i, (bad_patch, bad_mask) in enumerate(zip(slideImg.bad_patches, slideImg.bad_mask_patches)):
                                
                                filename = file_path + '\\%s_%d_%d.png' % (file_name, bad_patch[-2], bad_patch[-1])  
                                maskname = mask_path + '\\%s_%d_%d.jpg' % (file_name, bad_mask[-2], bad_mask[-1]) 
                                
                                result_BGR = cv2.cvtColor(bad_patch[0].squeeze(), cv2.COLOR_RGB2BGR)
                                                            
                                cv2.imwrite(filename, result_BGR)
                                cv2.imwrite(maskname, bad_mask[0].squeeze())
                                
                            print('Done')
                                
                        except Exception as ex:
                            print('Failed to process %s with error: %s' % (file, ex) )
                    
                        else:
                            continue
                        
                    elif patient_id in self.val_ids:
                                
                        try:
                            
                            slideImg = SlideImage(path = file, mask_path= mask, slide_level = self.slide_level, color_channels = "RGB" )
                            
                            file_path = self.target_dir + '\\Val\\img'
                            mask_path = self.mask_target_dir + '\\Val\\GT'
                            #os.makedirs(path, exist_ok = True)
                            
                            #id = slideImg.img_ID
                            
                            # if id not in img_ID_counts:
                            #     img_ID_counts[id] = 0
                            # else:
                            #     img_ID_counts[id] += 1
                                
                            print('Processing %s' % file_name)  
                            
                            if self.mode == 'PATCHES':
                                
                                slideImg.patchification(width=self.patch_size, height=self.patch_size, step=self.step)
                                
                            if self.use_mask:
                                
                                # threshold = slideImg.calc_lum_threshold( mode = self.filter_threshold, criteria = 0.1 )
                                # slideImg.generate_mask(threshold)
                                slideImg.mask_patchification(width=self.patch_size, height=self.patch_size, step=self.step)       
                            
                            slideImg.calc_border_patches()
                            #slideImg.crop_filter(self.crop_factor)
                            slideImg.variance_filter(self.variance_factor)
                            #slideImg.calc_border_patches() ## hace falta dos veces??
                            
                            slideImg.filter_patches()
                
                            for i, (patch, mask) in enumerate(zip(slideImg.filtered_patches, slideImg.filtered_mask_patches)):
                                
                                filename = file_path + '\\%s_%d_%d.png' % (file_name, patch[-2], patch[-1])  
                                maskname = mask_path + '\\%s_%d_%d.jpg' % (file_name, mask[-2], mask[-1])  
                                
                                result_BGR = cv2.cvtColor(patch[0].squeeze(), cv2.COLOR_RGB2BGR)
                                
                                cv2.imwrite(filename, result_BGR)
                                cv2.imwrite(maskname, mask[0].squeeze())
                            
                            for i, (bad_patch, bad_mask) in enumerate(zip(slideImg.bad_patches, slideImg.bad_mask_patches)):
                                
                                filename = file_path + '\\%s_%d_%d.png' % (file_name, bad_patch[-2], bad_patch[-1])  
                                maskname = mask_path + '\\%s_%d_%d.jpg' % (file_name, bad_mask[-2], bad_mask[-1]) 
                                
                                result_BGR = cv2.cvtColor(bad_patch[0].squeeze(), cv2.COLOR_RGB2BGR)
                                                            
                                cv2.imwrite(filename, result_BGR)
                                cv2.imwrite(maskname, bad_mask[0].squeeze())
                                
                            print('Done')
                                
                        except Exception as ex:
                            print('Failed to process %s with error: %s' % (file, ex) )
                    
                        else:
                            continue
                              

# %%
