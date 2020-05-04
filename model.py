import torch
from torch import nn
from torch.autograd import Variable

v_ker_size = 2
v_stride = 1

class CDAutoEncoder(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.

    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """
    def __init__(self, input_size, output_size, kernel_size, stride):
        super(CDAutoEncoder, self).__init__()

        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=2),
            nn.ReLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.ConvTranspose2d(output_size, input_size, kernel_size=kernel_size, stride=stride, padding=2), 
            nn.ReLU(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

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

        self.ae1 = CDAutoEncoder(1, 128, v_ker_size, v_stride)
        self.ae2 = CDAutoEncoder(128, 256, v_ker_size, v_stride)
        self.ae3 = CDAutoEncoder(256, 512, v_ker_size, v_stride)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)

        if self.training:
            return a3

        else:
            x_reconstruct, a1_reconstruct, a2_reconstruct = self.reconstruct(a3)
            return a1,a2,a3, x_reconstruct, a1_reconstruct, a2_reconstruct

    def reconstruct(self, x):
            a2_reconstruct = self.ae3.reconstruct(x)
            a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
            x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
            return x_reconstruct, a1_reconstruct, a2_reconstruct
