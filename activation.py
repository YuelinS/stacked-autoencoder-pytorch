# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:15:23 2020

@author: shiyl
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import glob

homefd = 'D:/git/stacked-autoencoder-pytorch/' #'D:/git/tmp5/' #
rfd = homefd + 'visualize/'
if not os.path.exists(rfd):
    os.makedirs(rfd)
    
datasz = '10k'
filenames = ['arr_0','arr_1','arr_2','arr_3','arr_4']
activations = np.load(homefd+'recon/batch_activation_' + datasz +'.npz')
#a0, a1, a2, a3, a4 = [activations[k] for k in filenames]
# (n_img, n_chn, sz, sz)

import scipy
import scipy.io
# 注意把layer 3和4换成更直观的顺序了
scipy.io.savemat(homefd+'recon/activations.mat',
                 {'layer_0': activations['arr_0'],'layer_1': activations['arr_1'],'layer_2': activations['arr_2'],
                  'layer_3': activations['arr_4'],'layer_4': activations['arr_3']})

saved_models = glob.glob(homefd+'saved_model_*.pth')
params = torch.load(saved_models[-1],map_location=torch.device('cpu'))


 

#%% visualize kernels in layer 0, each subplot is a kernel

layer = 0
row, col = 2, 8  # 1,3  #

ker_ff = params['ae'+str(layer)+'.forward_pass.0.weight'].numpy()
ker_fb = params['ae'+str(layer)+'.backward_pass.0.weight'].numpy()
layer_dict = {'ff':ker_ff,'fb':ker_fb}

n_chn = ker_ff.shape[0]
flag_chn = range(min(row*col,n_chn))  #range(n_chn)   # [0]  #
    

for channel in flag_chn:
        
        iax = channel % (row * col)

        if iax == 0:
        
            plt.close('all')   
            fig, axs = plt.subplots(row*2, col, figsize=(20, 12))  #, facecolor='w', edgecolor='k')
            # fig2.subplots_adjust(hspace = .5, wspace=.001)
            axs = axs.ravel()            
            ifig = channel // ( row * col)

        for i,type in enumerate(layer_dict):
            iiax = row * col * i + iax
            axs[iiax].imshow(layer_dict[type][channel,0], cmap='gray')
            axs[iiax].set_title(f'Neuron {channel}')            
            
        if iax == (row * col)-1 or channel == len(flag_chn)-1:
    
            axe = fig.add_subplot(211, frameon=False)
            axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            axe.set_ylabel('Feedforward',position = [.5, 1.05])    
            
            axe = fig.add_subplot(212, frameon=False)
            axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            axe.set_ylabel('Feedback',position = [.5, 1.05])
            
            fig.subplots_adjust(hspace = 0.4)
            fig.savefig(rfd + f'kernels_layer{layer}_chns#{ifig}.png')


#%% visualize kernels in layer > 0, each row is a kernel, each row is the kernels inner channels

layer = 1
row, col = 3, 9

ker_ff = params['ae'+str(layer)+'.forward_pass.0.weight'].numpy()
ker_fb = params['ae'+str(layer)+'.backward_pass.0.weight'].numpy()
layer_dict = {'ff':ker_ff,'fb':ker_fb}

n_chn = ker_ff.shape[0]
flag_chn = range(min(row*2,n_chn))   # [0]  #range(16)  #
    

for channel in flag_chn:
        
        irow = channel % row

        if irow == 0:
        
            plt.close('all')   
            fig, axs = plt.subplots(row*2, col, figsize=(20, 12))  #, facecolor='w', edgecolor='k')
            # fig2.subplots_adjust(hspace = .5, wspace=.001)
            axs = axs.ravel()            
            ifig = channel // row 

        for i,type in enumerate(layer_dict):
            iirow = row * i + irow
            for icol in range(col):
                iax = (iirow-1) * col + icol
                axs[iax].imshow(layer_dict[type][channel,icol], cmap='gray')
                axs[iax].set_title(f'Kernel{channel} channel{icol}')            
            
        if irow == row-1 or channel == len(flag_chn)-1:
    
            axe = fig.add_subplot(211, frameon=False)
            axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            axe.set_ylabel('Feedforward',position = [.5, 1.05])    
            
            axe = fig.add_subplot(212, frameon=False)
            axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            axe.set_ylabel('Feedback',position = [.5, 1.05])
            
            fig.subplots_adjust(hspace = 0.4)
            fig.savefig(rfd + f'kernels_layer{layer}_chns#{ifig}.png')



#%% show activation in layer 0, show multiplu images, each only has a single channel

layer = 0
ff = activations[filenames[layer]]
fb = activations[filenames[layer+3]]
layer_dict = {'ff':ff,'fb':fb}

row, col = 2, 8 
flag_img = min(ff.shape[0],row*col)


for img in range(flag_img):           
    
        
        iax = img % (row * col)

        if iax == 0:
        
            plt.close('all')   
            fig, axs = plt.subplots(row*2, col, figsize=(20, 12))  #, facecolor='w', edgecolor='k')
            # fig2.subplots_adjust(hspace = .5, wspace=.001)
            axs = axs.ravel()            
            ifig = img // ( row * col)

        for i,type in enumerate(layer_dict):
            iiax = row * col * i + iax
            axs[iiax].imshow(layer_dict[type][img,0], cmap='gray')
            axs[iiax].set_title(f'Img {img}')            
            
        if iax == (row * col)-1 or img == flag_img-1:
    
            axe = fig.add_subplot(211, frameon=False)
            axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            axe.set_ylabel('Feedforward',position = [.5, 1.05])    
            
            axe = fig.add_subplot(212, frameon=False)
            axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            axe.set_ylabel('Feedback',position = [.5, 1.05])
            
            fig.subplots_adjust(hspace = 0.4)
            fig.savefig(rfd + f'pop_activation_layer{layer}.png')



#%% show activation in layer>0, for each image, show multiple channels
layer = 2

ff = activations[filenames[layer]]
if layer == 2:
    layer_dict = {'ff':ff}
else:
    layer_dict = {'ff':ff,'fb':fb}
    fb = activations[filenames[layer+3]]


## For each image
row, col = 3, 9  # 1,3  #
flag_img = range(min(2,ff.shape[0]))  #range(batch_size)  # 
flag_chn = range(min(row*col,ff.shape[1]))  # range(64)  #[0]  #
                 
                 
for img in flag_img:        
        
    for channel in flag_chn:
        
        iax = channel % (row * col)

        if iax == 0:
        
            plt.close('all')   
            fig, axs = plt.subplots(row*2, col, figsize=(20, 12))  #, facecolor='w', edgecolor='k')
            # fig2.subplots_adjust(hspace = .5, wspace=.001)
            axs = axs.ravel()            
            ifig = channel // ( row * col)

        for i,type in enumerate(layer_dict):
            iiax = row * col * i + iax
            axs[iiax].imshow(layer_dict[type][img,channel], cmap='gray')
            axs[iiax].set_title(f'Neuron {channel}')            
            
        if iax == (row * col)-1 or channel == len(flag_chn)-1:
    
            axe = fig.add_subplot(211, frameon=False)
            axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            axe.set_ylabel('Feedforward',position = [.5, 1.05])    
            
            axe = fig.add_subplot(212, frameon=False)
            axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            axe.set_ylabel('Feedback',position = [.5, 1.05])
            
            fig.subplots_adjust(hspace = 0.4)
            fig.savefig(rfd + f'pop_activation_layer{layer}_img{img}_chns#{ifig}.png')


#%% For each neuron/channel, at one location, show it's ff/fb response across images
           
flag_chn = range(n_chn)  # range(3)  # [0]  #
row, col = 4, 4  # 1,3  #


sz = ff.shape[2]
for channel in flag_chn:
    
    for pos_x, pos_y in zip([8],[8]): # range(sz):
        
        iax = channel % (row * col)
    
        if iax == 0:
        
            plt.close('all')   
            fig, axs = plt.subplots(row, col, figsize=(20, 12))  #, facecolor='w', edgecolor='k')
            # fig2.subplots_adjust(hspace = .5, wspace=.001)
            axs = axs.ravel()            
            ifig = channel // ( row * col)

        axs[iax].plot(ff[:,channel, pos_x, pos_y])           
        axs[iax].plot(fb[:,channel, pos_x, pos_y])
        axs[iax].set_title(f'Neuron {channel}')
        
        if iax == (row * col)-1 or channel == len(flag_chn)-1:
    
            axe = fig.add_subplot(111, frameon=False)
            axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            axe.set_title(f'Activity to position {pos_x, pos_y}',position = [.5, 1.05])    
            
                
            fig.subplots_adjust(hspace = 0.4)
            fig.savefig(rfd + f'sin_activation_layer{layer}_chns#{ifig}.png')

        
                    
            