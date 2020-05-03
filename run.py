import os
import time
import tqdm
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from model import StackedAutoEncoder
from FaceDataset import FaceDataset

num_epochs = 100  # 1000
batch_size = 200 # 128
flag_sample_epoch = 20

data_path = 'D:/180921_Feedback/5Occluded_AM/stim2/prepare_real_Liang/human_face_2000/2000' # 'D:/git/imgs/'
size = 64

def to_img(x):
    x = x.view(x.size(0), 3, size, size)
    return x


img_transform = transforms.Compose([
    #transforms.RandomRotation(360),
#    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
    transforms.Resize((64,46)),
    transforms.Pad((9,0,9,0), fill=0, padding_mode='constant'),
    transforms.ToTensor(),
])

dataset = FaceDataset(data_path, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = StackedAutoEncoder().cuda()

for epoch in range(num_epochs):
    # if epoch % 10 == 0:
    #     # Test the quality of our features with a randomly initialzed linear classifier.
    #     classifier = nn.Linear(512 * 16, 10).cuda()
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    model.train()
    total_time = time.time()
    correct = 0
    for i, img in enumerate(dataloader):  
        img = Variable(img).cuda()
        features = model(img).detach()
        # prediction = classifier(features.view(features.size(0), -1))
        # loss = criterion(prediction, target)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # pred = prediction.data.max(1, keepdim=True)[1]
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    total_time = time.time() - total_time

    model.eval()
    features, x_reconstructed = model(img)
    reconstruction_loss = torch.mean((x_reconstructed.data - img.data)**2)

    if epoch % flag_sample_epoch == 0:
        print("Saving epoch {}".format(epoch))
        orig = to_img(img.cpu().data)
        save_image(orig, '../recon/orig_{}.png'.format(epoch))
        pic = to_img(x_reconstructed.cpu().data)
        save_image(pic, '../recon/reconstruction_{}.png'.format(epoch))

    print("Epoch {} complete\tTime: {:.4f}s\t\tLoss: {:.4f}".format(epoch, total_time, reconstruction_loss))
    print("Feature Statistics\tMean: {:.4f}\t\tMax: {:.4f}\t\tSparsity: {:.4f}%".format(
        torch.mean(features.data), torch.max(features.data), torch.sum(features.data == 0.0)*100 / features.data.numel())
    )
    # print("Linear classifier performance: {}/{} = {:.2f}%".format(correct, len(dataloader)*batch_size, 100*float(correct) / (len(dataloader)*batch_size)))
    print("="*80)

# torch.save(model.state_dict(), './CDAE.pth')
