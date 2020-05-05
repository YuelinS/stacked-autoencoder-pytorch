# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:15:23 2020

@author: shiyl
"""
import numpy as np
import matplotlib.pyplot as plt

rfd = 'D:/git/visualize/'
datasz = '10k'


filenames = ['arr_0','arr_1','arr_2','arr_3','arr_4']
activations = np.load('../recon/batch_activation_' + datasz +'.npz')
a0, a1, a2, a3, a4 = [activations[k] for k in filenames]

# (n_img, n_chn, sz, sz)

#%% 
layer = 0
ff = activations[filenames[layer]]
fb = activations[filenames[layer+3]]
n_img = ff.shape[0]
row, col = 4, 4 
layer_dict = {'ff':ff,'fb':fb}

for img in range(n_img):           
    
        
        iax = img % (row * col)

        if iax == 0:
        
            plt.close('all')   
            fig, axs = plt.subplots(row, col*2, figsize=(20, 12))  #, facecolor='w', edgecolor='k')
            # fig2.subplots_adjust(hspace = .5, wspace=.001)
            axs = axs.ravel()            
            ifig = img // ( row * col)

        for i,type in enumerate(layer_dict):
            iiax = row * col * i + iax
            axs[iiax].imshow(layer_dict[type][img,0], cmap='gray')
            axs[iiax].set_title(f'Img {img}')            
            
        if iax == (row * col)-1 or img == n_img-1:
    
            axe = fig.add_subplot(211, frameon=False)
            axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            axe.set_ylabel('Feedforward',position = [.5, 1.05])    
            
            axe = fig.add_subplot(212, frameon=False)
            axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            axe.set_ylabel('Feedback',position = [.5, 1.05])
            
            fig.subplots_adjust(hspace = 0.4)
            fig.savefig(rfd + f'pop_activation_layer{layer}.png')



#%%
layer = 1

ff = activations[filenames[layer]]
fb = activations[filenames[layer+3]]
n_chn = ff.shape[1]


## For each image

flag_img = range(2)  #range(batch_size)  # 
flag_chn = range(n_chn)  #range(20)  # [0]  #
row, col = 4, 4  # 1,3  #


layer_dict = {'ff':ff,'fb':fb}
n_chn = ff.shape[1]
for img in flag_img:           
        
    for channel in flag_chn:
        
        iax = channel % (row * col)

        if iax == 0:
        
            plt.close('all')   
            fig, axs = plt.subplots(row, col*2, figsize=(20, 12))  #, facecolor='w', edgecolor='k')
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


## For each neuron
           
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

        
                    
            