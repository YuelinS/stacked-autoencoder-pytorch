import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau 

n_layer = 2
# n_chn = [1, 80, 80, 256]
n_chn = [1, 60, 10]
output_paddings = [0,1,0]
v_ker_size = 4 
v_stride = 2
lr = 1e-3


class CDAutoEncoder(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.

    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """
    def __init__(self, input_size, output_size, kernel_size, stride,output_padding):
        super(CDAutoEncoder, self).__init__()
        
        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=0),
            nn.ReLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.ConvTranspose2d(output_size, input_size, kernel_size=kernel_size, stride=stride, padding=0,output_padding=output_padding), 
            nn.ReLU(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.3, patience=5)

        
        
    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()
        # Add noise, but use the original lossless input as the target.
        # x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        # y = self.forward_pass(x_noisy)
        y = self.forward_pass(x)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)


class StackedAutoEncoder(nn.Module):
    r"""
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self):
        super(StackedAutoEncoder, self).__init__()
        for layer in range(n_layer):
            self.add_module(f'ae{layer}',CDAutoEncoder(n_chn[layer], n_chn[layer+1], v_ker_size, v_stride,output_paddings[layer]))


    def forward(self, x):
        fw_out =  Variable(x).cuda()
        fw_outs = []
        for layer in range(n_layer):
            fw_in =  Variable(fw_out).cuda()
            fw_out = self._modules[f'ae{layer}'](fw_in)
            fw_outs.append(fw_out)


        if self.training:
            return fw_out

        else:

            rc_outs = self.reconstruct(fw_out)
            return fw_outs + rc_outs


    def reconstruct(self, x):
        rc_out = x
        rc_outs = []
        for layer in reversed(range(n_layer)):
            rc_in = rc_out
            rc_out = self._modules[f'ae{layer}'].reconstruct(rc_in)
            rc_outs.insert(0,rc_out)
        return rc_outs
        
    
    def update_scheduler(self,recon_loss): 
        for layer in range(n_layer):
            self._modules[f'ae{layer}'].scheduler.step(recon_loss)
