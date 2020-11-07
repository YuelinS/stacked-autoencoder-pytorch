# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:15:49 2020

@author: shiyl
"""

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#
    
#from __future__ import print_function, division
import os
import torch
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
        
        
class FaceArrayDataset(Dataset):
    """Face dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        # import pdb; pdb.set_trace()        
        self.array = np.load(self.root_dir,allow_pickle = True)           
        self.transform = transform

    def __len__(self):       
        return len(self.array)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = Image.fromarray(self.array[idx])      

        if self.transform:
            sample = self.transform(sample)

        return sample    

    
#%%
class FaceDataset(Dataset):
    """Face dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.file_names = [f for f in os.listdir(root_dir) if '.png' or '.jpg' in f]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.file_names[idx])
        sample = Image.fromarray(io.imread(img_name))      

        if self.transform:
            sample = self.transform(sample)

        return sample    
    
          

#%% Take a look at the first 3 images
# =============================================================================
# import matplotlib.pyplot as plt
# from torchvision import transforms
# 
# data_path = 'D:/git/stacked-autoencoder-pytorch/imgs/face_array_10k.npy' # 'D:/180921_Feedback/5Occluded_AM/stim2/prepare_real_Liang/human_face_2000/2000'  #'D:/git/imgs/' #  
# 
# 
# img_transform = transforms.Compose([
#     #transforms.RandomRotation(360),
# #    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
#     transforms.Resize(32),
#     transforms.Pad((6,0,7,0), fill=0, padding_mode='constant'),
#     transforms.ToTensor(),
# ])
# 
# face_dataset = FaceArrayDataset(data_path, transform=img_transform)
# 
# 
# fig = plt.figure()
# 
# for i in range(3):
#     sample = face_dataset[i]
# 
#     print(i, sample.shape)
# 
#     ax = plt.subplot(1, 3, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     # in pytorch uses gbr? this model allows color img
#     plt.imshow(sample.numpy().transpose(1,2,0)[:,:,0])   
#     plt.show()
# =============================================================================
  