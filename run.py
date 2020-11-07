import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
# import sys
# os.chdir('D:/git/stacked-autoencoder-pytorch')
# sys.path.append('D:/git/stacked-autoencoder-pytorch')
from model import StackedAutoEncoder
from FaceDataset import FaceArrayDataset
import numpy as np
from datetime import datetime
import random


seed = 42 #42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
    
    
homefd = 'D:/git/stacked-autoencoder-pytorch/'  #'D:/git/tmp5/'
reconfd = homefd+'recon/'
if not os.path.exists(reconfd):
    os.makedirs(reconfd)


num_epochs = 12   # 1 # 100
batch_size = 64 # 128
flag_sample_epoch = 10

datasz = '10k'  # '100k'
data_path = 'D:/git/stacked-autoencoder-pytorch/imgs/face_array_' + datasz + '.npy' # 'D:/180921_Feedback/5Occluded_AM/stim2/prepare_real_Liang/human_face_2000/2000' # 'D:/git/imgs/'
size = 32
n_chn = 1


def to_img(x):
    x = x.view(x.size(0), n_chn, size, size)
    return x

# def main():
img_transform = transforms.Compose([
    #transforms.RandomRotation(360),
#    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
    transforms.Resize((32,22)),  # 64,45
    transforms.Pad((5,0,5,0), fill=128/255, padding_mode='constant'),  # 9,0,10,0
    transforms.ToTensor(),
])

dataset = FaceArrayDataset(data_path, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = StackedAutoEncoder().cuda()
#    model = model.to('cuda')

for epoch in range(num_epochs):

    model.train()
    total_time = time.time()
    for i, img in enumerate(dataloader):  
        img = Variable(img).cuda()
        model(img).detach()

    total_time = time.time() - total_time

    model.eval()
#        _,_,features, x_recon,_,_ = model(img)
    _,features, x_recon,_ = model(img)
    recon_loss = torch.mean((x_recon.data - img.data)**2)
    
    model.update_scheduler(recon_loss)
    
    # validate


    # save a batch of images and recon every xxx epochs
    if epoch % flag_sample_epoch == 0:
        print("Saving epoch {}".format(epoch))
        orig = to_img(img.cpu().data)
        save_image(orig, homefd+'recon/orig_{}.png'.format(epoch))
        pic = to_img(x_recon.cpu().data)
        save_image(pic, homefd+'recon/recon_{}.png'.format(epoch))

    # print progress after every epoch
    print("Epoch {} complete\tTime: {:.4f}s\t\tLoss: {:.4f}".format(epoch, total_time, recon_loss))
    print("Feature Statistics\tMean: {:.4f}\t\tMax: {:.4f}\t\tSparsity: {:.4f}%".format(
        torch.mean(features.data), torch.max(features.data), torch.sum(features.data == 0.0)*100 / features.data.numel()))
    # print("Linear classifier performance: {}/{} = {:.2f}%".format(correct, len(dataloader)*batch_size, 100*float(correct) / (len(dataloader)*batch_size)))
    print("="*80)

    if epoch == num_epochs-1:
#       a1,a2,features, x_recon, a1_recon, a2_recon = model(img)
        outputs_ts = model(img)
        outputs = [img.detach().cpu().numpy()] + [k.detach().cpu().numpy() for k in outputs_ts] 
        np.savez(homefd+'recon/batch_activation_' + datasz + '.npz', *outputs)

now = datetime.now()
dt_string = now.strftime("%m%d_%H%M")
torch.save(model.state_dict(), homefd+'saved_model_'+dt_string+'.pth')


# if __name__ == '__main__':
#     main()