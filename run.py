import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from model import StackedAutoEncoder
from FaceDataset import FaceArrayDataset
import numpy as np

num_epochs = 1  # 1000
batch_size = 64 # 128
flag_sample_epoch = 6

datasz = '10k'  # '100k'
data_path = 'D:/git/imgs/face_array_' + datasz + '.npy' # 'D:/180921_Feedback/5Occluded_AM/stim2/prepare_real_Liang/human_face_2000/2000' # 'D:/git/imgs/'
size = 32
n_chn = 1


def to_img(x):
    x = x.view(x.size(0), n_chn, size, size)
    return x

def main():
    img_transform = transforms.Compose([
        #transforms.RandomRotation(360),
    #    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
        transforms.Resize((32,22)),
        transforms.Pad((5,0,5,0), fill=128/255, padding_mode='constant'),
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
        _,_,features, x_recon,_,_ = model(img)
        recon_loss = torch.mean((x_recon.data - img.data)**2)
    
        model.update_scheduler(recon_loss)
    
        # save a batch of images and recon every xxx epochs
        if epoch % flag_sample_epoch == 0:
            print("Saving epoch {}".format(epoch))
            orig = to_img(img.cpu().data)
            save_image(orig, '../recon/orig_{}.png'.format(epoch))
            pic = to_img(x_recon.cpu().data)
            save_image(pic, '../recon/recon_{}.png'.format(epoch))
    
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
            np.savez('../recon/batch_activation_' + datasz + '.npz', *outputs)
    
         
    # torch.save(model.state_dict(), './CDAE.pth')

if __name__ == '__main__':
    main()