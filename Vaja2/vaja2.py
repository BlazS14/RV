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
def generate_convex_quadrilateral_points(min, max):
    while True:
        x1, y1 = random.randint(min, max), random.randint(min, max)
        x2, y2 = random.randint(min, max), random.randint(min, max)
        x3, y3 = random.randint(min, max), random.randint(min, max)
        x4, y4 = random.randint(min, max), random.randint(min, max)
        cross_product1 = (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1)
        cross_product2 = (x3 - x2) * (y4 - y3) - (x4 - x3) * (y3 - y2)
        cross_product3 = (x4 - x3) * (y1 - y4) - (x1 - x4) * (y4 - y3)
        cross_product4 = (x1 - x4) * (y2 - y1) - (x2 - x1) * (y1 - y4)
        if cross_product1 > 0 and cross_product2 > 0 and cross_product3 > 0 and cross_product4 > 0:
            return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
def create_image(height, width):
    img = np.zeros((height,width,3), np.uint8)
    img[:,:] = [random.randint(0, 256),random.randint(0, 256),random.randint(0, 256)]
    return img
def generate_rectangle(img):
    height, width = img.shape[:2]
    # generate an image and a random rectangle
    points = generate_convex_quadrilateral_points(height,width)
    cv2.drawContours(img, [points], 0, (random.randint(0, 256),random.randint(0, 256),random.randint(0, 256)), -1)
    return img, points
def generate_triangle(img):
    height, width = img.shape[:2]
    p1 = [random.randint(0, height), random.randint(0, width)]
    p2 = [random.randint(0, height), random.randint(0, width)]
    p3 = [random.randint(0, height), random.randint(0, width)]
    cv2.drawContours(img, [np.array([p1,p2,p3])], 0, (random.randint(0, 256),random.randint(0, 256),random.randint(0, 256)), -1)
    return img,np.array([p1,p2,p3])
def generate_grid(img):
    '''height, width = img.shape[:2]
    #generate a grid with random colors and save its points in an array
    points = generate_convex_quadrilateral_points(height,width)
    xnummax = min((points[0,0]-points[1,0],points[3,0]-points[2,0]))
    ynummax = min((points[0,1]-points[3,1],points[1,1]-points[2,1]))
    if ynummax > 10 and xnummax > 10:
        counter = 0
        while counter != 10:
            xnum = random.randint(2, 5)
            ynum = random.randint(2, 5)
            if ynummax % ynum == 0 and xnummax % xnum == 0:
                break
        if counter == 10:
            xnum = 0
            ynum = 0
    else:
        xnum = 0
        ynum = 0
    xinc = xnummax/xnum
    yinc = ynummax/ynum'''
def generate_star(img):
    height, width = img.shape[:2]
    point_array = np.empty((0,2), dtype=np.int32)
    center = [random.randint(0, height), random.randint(0, width)]
    #generate 4 to 5 random points around center and connect them to the center
    point_array = np.append(point_array, [center], axis=0)
    for i in range(random.randint(4,5)):
        point = [random.randint(0, height), random.randint(0, width)]
        point_array = np.append(point_array, [point], axis=0)
        cv2.line(img, center, point, (random.randint(0, 256),random.randint(0, 256),random.randint(0, 256)), 1)
    return img, point_array
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
        x = random.randint(0+40,320-40)
        y = random.randint(0+40,240-40)
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
disc_model = Discriminator()
disc_optim = torch.optim.Adam(disc_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_loss_fn = nn.BCELoss()
gen_model = Generator()
gen_optim = torch.optim.Adam(gen_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
gen_loss_fn = nn.BCELoss()
data_set = ImageDataSet()
data_gen = DataLoader(data_set, batch_size=128,shuffle=True)
train(disc_model, disc_optim, disc_loss_fn, gen_model, gen_optim, gen_loss_fn,
      1000, 128, data_gen,create_identity(1))
torch.save(disc_model, "model_Discriminator.pt")
torch.save(gen_model, "model_Generator.pt")