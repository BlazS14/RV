import cv2
import tkinter as tk
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import os
import random
from torch.utils.data import Dataset, DataLoader
from math import log10, sqrt
from skimage.metrics import structural_similarity as SSIM
import torchinfo
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
def calculate_frame_offsets(dirpath):
    offests = np.empty((0,3), dtype=np.int32)
    dirlist = os.listdir(dirpath)
    for n in range(len(dirlist)):
                offests = np.append(offests,[[n]],axis=0)
    return offests
def get_images(dirpath):
    files = []
    for file in os.listdir(dirpath):
        files.append(dirpath+file)
    return files
class ImageDataSet(Dataset):
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.offsets = calculate_frame_offsets(dirpath)
    def __len__(self):
        return len(self.offsets)
    def __getitem__(self, index):
        dirlist = os.listdir(self.dirpath)
        img = cv2.imread(self.dirpath+dirlist[self.offsets[index][0]])
        img = cv2.resize(img, 320, 240)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = random.randint(0+48,320-48)
        y = random.randint(0+48,240-48)
        corners = np.array([[x-32,y-32],[x+32,y-32],[x+32,y+32],[x-32,y+32]], dtype=np.int32)
        transformed_corners = np.array([[x-32+random.randint(-16,16),y-32+random.randint(-16,16)],[x+32+random.randint(-16,16),y-32+random.randint(-16,16)],[x+32+random.randint(-16,16),y+32+random.randint(-16,16)],[x-32+random.randint(-16,16),y+32+random.randint(-16,16)]], dtype=np.int32)
        homography_matrix, _ = cv2.findHomography(corners,transformed_corners)
        transformed_img = cv2.warpPerspective(img, homography_matrix, (320,240))
        img = np.concatenate((img,transformed_img),axis=1)
        img = img/255
        return np.array(img).astype(np.float32),corners,transformed_corners,homography_matrix
    
    
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        # If the input and output channels don't match, we need to apply a 1x1 convolution to match the dimensions.
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
class HomoNet(nn.Module):
    def __init__(self):
        super(HomoNet, self).__init__()
        self.resnet1 = ResNet(2, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.resnet2 = ResNet(64, 64)
        self.linear1 = nn.Linear(64,512)
        self.resnet3 = ResNet(64, 128)
        self.resnet4 = ResNet(128, 128)
        self.regression_head = nn.Linear(128, 8)
    def forward(self, x):
        x = self.resnet1(x)
        x = self.resnet2(x)
        x = self.maxpool1(x)
        x = self.resnet2(x)
        x = self.resnet2(x)
        x = self.maxpool1(x)
        x = self.resnet3(x)
        x = self.resnet4(x)
        x = self.maxpool1(x)
        x = self.resnet4(x)
        x = self.resnet4(x)
        x = self.linear1(x)
        x = self.regression_head(x)
        return x
class ImageDataSet(Dataset):
    def __init__(self):
        mnist_ds = torchvision.datasets.MNIST(
        root="datasets", train=True, transform=torchvision.transforms.ToTensor(),
        download=True)
        print(f"Širina slik: {mnist_ds[0][0].shape[2]}")
        print(f"Višina slik: {mnist_ds[0][0].shape[1]}")
        print(f"Število kanalov: {mnist_ds[0][0].shape[0]}")
        print(f"Število slik: {len(mnist_ds)}")
        print(f"Podatkovna zbirka: MNIST (http://yann.lecun.com/exdb/mnist/)")
        n_rows = 5
        n_cols = 5
        _, axes = plt.subplots(n_rows, n_cols)
        for r in range(n_rows):
            for c in range(n_cols):
                img, _ = mnist_ds[random.randint(0, len(mnist_ds) - 1)]
                axes[r, c].imshow(img.permute(1, 2, 0), cmap="gray")
                axes[r, c].axis("off")
        plt.tight_layout()
        plt.show()
        self.images = mnist_ds
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        img, _ = self.images[index]
        img = img/255
        return img
def create_identity(num):
    identity_array = np.empty((0,100), dtype=np.float32)
    for i in range(num):
        identity = np.random.standard_normal(100)
        identity = (identity + abs(np.min(identity)))
        identity = identity / np.max(identity)
        identity_array = np.append(identity_array, [identity], axis=0)
    identity_array = torch.from_numpy(identity_array.astype(np.float32))
    return identity_array
def train(discriminator_model, discriminator_optimizer, discriminator_loss_fn, generator_model, generator_optimizer, generator_loss_fn, epochs, batch_size,dataset,static_identity):
    discriminator_loss_sum = 0
    generator_loss_sum = 0
    for epoch in range(1,epochs+1):
        discriminator_optimizer.zero_grad()
        data = next(iter(dataset))
        discriminator_output = discriminator_model(data)
        discriminator_loss = discriminator_loss_fn(discriminator_output, torch.ones(batch_size,1))
        discriminator_loss.backward()
        generator_output = generator_model(create_identity(batch_size))
        discriminator_output = discriminator_model(generator_output)
        discriminator_loss = discriminator_loss_fn(discriminator_output, torch.zeros(batch_size,1))
        discriminator_loss.backward()
        discriminator_optimizer.step()
        img = generator_model(static_identity)
        img = img * 255
        img = img.detach().numpy()[0,0,:,:]
        img = img.astype(np.uint8)
        discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()
        generator_model.zero_grad()
        generator_output = generator_model(create_identity(batch_size))
        discriminator_output = discriminator_model(generator_output)
        generator_loss = generator_loss_fn(discriminator_output, torch.ones(batch_size,1))
        generator_loss.backward()
        generator_optimizer.step()
        generator_loss_sum += generator_loss.item()
        discriminator_loss_sum += discriminator_loss.item()
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Generator loss: {generator_loss_sum/10}, Discriminator loss: {discriminator_loss_sum/10}")
            generator_loss_sum = 0
            discriminator_loss_sum = 0
            cv2.imwrite("img"+str(epoch)+".png",img)

model = HomoNet()
#model.load_state_dict(torch.load("model_3380.pth"))
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
data_set = ImageDataSet("C:\\Users\\GTAbl\\Desktop\\RV\\Vaja3\\data\\train")
loss_fn = EPELoss()
data_gen = DataLoader(data_set, batch_size=8,shuffle=True)
train(model, loss_fn, optim, 500000, data_gen)