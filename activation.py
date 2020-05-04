# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:15:23 2020

@author: shiyl
"""
import numpy as np
import matplotlib.pyplot as plt

rfd = 'D:/git/visualize'

batch_size = 200
filenames = ['arr_0','arr_1','arr_2','arr_3','arr_4','arr_5']

activations = np.load('../recon/batch_activation.npz')
a1, a2, features, x_recon, a1_recon, a2_recon = [activations[k] for k in filenames]


#%% For each image

row = 3
col = 5

for layer in range(1):
    orig = activations[filenames[layer-1]]
    recon = activations[filenames[layer-1+4]]
    layer_dict = {'orig':orig,'recon':recon};
    n_chn = orig.shape[1]
        
    for img in range(batch_size):    
    
        for type in layer_dict:
            
            for channel in range(n_chn):
                
                iax = channel % (row * col)
        
                if iax == 0:
                
                    plt.close('all')   
                    fig, axs = plt.subplots(row, col, figsize=(20, 12))  #, facecolor='w', edgecolor='k')
                    # fig2.subplots_adjust(hspace = .5, wspace=.001)
    #                axs = axs.ravel()            
                    ifig = channel // ( row * col)
    
                axs[iax].imshow(layer_dict[type][img,channel])
                
                if iax == (row * col)-1 or channel == n_chn-1:
            
                    axe = fig.add_subplot(111, frameon=False)
                    axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                    axe.set_title(f'Layer #{layer}  Img #{img}',position = [.5, 1.05])    
                    
                        
                    fig.subplots_adjust(hspace = 0.4)
                    fig.savefig(rfd + f'pop_activation_layer{layer}_img{img}_channels{ifig}_' + type + '.png')


#%% For each neuron
                
for layer in range(1):
    orig = activations[filenames[layer-1]]
    recon = activations[filenames[layer-1+4]]
    n_chn = orig.shape[1]
    sz = orig.shape[2]
    
    for channel in range(n_chn):
        
        for pos_x, pos_y in zip(range(20,20),range(20,20)): #in range(sz*sz):
            
            iax = channel % (row * col)
        
            if iax == 0:
            
                plt.close('all')   
                fig, axs = plt.subplots(row, col, figsize=(20, 12))  #, facecolor='w', edgecolor='k')
                # fig2.subplots_adjust(hspace = .5, wspace=.001)
#                axs = axs.ravel()            
                ifig = channel // ( row * col)

            axs[iax].plot(orig[:,channel, pos_x, pos_y])
            
            axs[iax].plot(recon[:,channel, pos_x, pos_y])
            
            if iax == (row * col)-1 or channel == n_chn-1:
        
                axe = fig.add_subplot(111, frameon=False)
                axe.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                axe.set_title(f'Layer #{layer}  Position {pos_x, pos_y}',position = [.5, 1.05])    
                
                    
                fig.subplots_adjust(hspace = 0.4)
                fig.savefig(rfd + f'sin_activation_layer{layer}_channels{ifig}.png')

        
        
            
            