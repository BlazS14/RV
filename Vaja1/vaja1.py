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

import torch_directml
dml = torch_directml.device()
torch.set_default_device(dml)
torch.set_default_dtype(torch.float32)



#write numpy array to file
def write_to_file(array, filename):
    with open(filename, 'wb') as f:
        np.save(f, array)
        
#read numpy array from file
def read_from_file(filename):
    with open(filename, 'rb') as f:
        array = np.load(f)
    return array

#check if file exists
def file_exists(filename):
    return os.path.isfile(filename)

def calculate_frame_offsets(dirpath):
    offests = np.empty((0,1), dtype=np.int32)
    dirlist = os.listdir(dirpath)
    for n in range(len(dirlist)):
                offests = np.append(offests,[[dirlist[n]]],axis=0)
    return offests
def get_images(dirpath):
    files = []
    for file in os.listdir(dirpath):
        files.append(dirpath+file)
    return files
class ImageDataSet(Dataset):
    def __init__(self, dirpath):
        self.dirpath = dirpath
        if not file_exists(dirpath+"\\offsets.npy"):
            self.offsets = calculate_frame_offsets(dirpath)
            write_to_file(self.offsets, dirpath+"\\offsets.npy")
        else:
            self.offsets = read_from_file(dirpath+"\\offsets.npy")
    def __len__(self):
        return len(self.offsets)
    def __getitem__(self, index):
        path = self.dirpath+"\\"+self.offsets[index][0]
        img = cv2.imread(path)
        img = cv2.resize(img, (320, 240))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orig_img = img.copy()
        x = random.randint(0+48,320-48)
        y = random.randint(0+48,240-48)
        corners = np.array([[x-32,y-32],[x+32,y-32],[x+32,y+32],[x-32,y+32]], dtype=np.int32)
        transformed_corners = np.array([[x-32+random.randint(-16,16),y-32+random.randint(-16,16)],[x+32+random.randint(-16,16),y-32+random.randint(-16,16)],[x+32+random.randint(-16,16),y+32+random.randint(-16,16)],[x-32+random.randint(-16,16),y+32+random.randint(-16,16)]], dtype=np.int32)
        homography_matrix, _ = cv2.findHomography(corners,transformed_corners)
        transformed_img = cv2.warpPerspective(img, homography_matrix, (320,240),flags=cv2.WARP_INVERSE_MAP)
        
        transformed_img = transformed_img[y-32:y+32,x-32:x+32]
        
        corner_offsets = transformed_corners - corners
        corner_offsets = corner_offsets.flatten()
        corner_offsets = corner_offsets.astype(np.float32)
        corner_offsets += 16
        corner_offsets /= 32
        
        img = img[y-32:y+32,x-32:x+32]
        
        '''cv2.imshow("img",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("img",transformed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
        img = cv2.merge((img,transformed_img))
        img = img/255
        return np.array(img).astype(np.float32),corners,transformed_corners,homography_matrix,corner_offsets,orig_img

def show_corners_on_image(img,orig_corners,transformed_corners, model_corners):
    orig_img = img
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    orig_corners = orig_corners.astype(np.int32)
    transformed_corners = transformed_corners.astype(np.int32)
    model_corners = model_corners.astype(np.int32)
    
    cv2.rectangle(orig_img, (orig_corners[0,0],orig_corners[0,1]), (orig_corners[3,0],orig_corners[3,1]), (255,0,0), 1)
    cv2.rectangle(orig_img, (transformed_corners[0,0],transformed_corners[0,1]), (transformed_corners[3,0],transformed_corners[3,1]), (0,255,0), 1)
    cv2.rectangle(orig_img, (model_corners[0,0],model_corners[0,1]), (model_corners[3,0],model_corners[3,1]), (0,0,255), 1)
    cv2.imshow("img",orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
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
        self.resnet3 = ResNet(64, 128)
        self.resnet4 = ResNet(128, 128)
        self.linear1 = nn.Linear(8192,512)
        self.regression_head = nn.Linear(512, 8)
        
        self.cls_linear1 = nn.Linear(128*8*8, 8*21)
        self.cls_softmax = nn.Softmax(dim=2)
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
        
        #x = self.avgpool(x)  
        x = torch.flatten(x, 1) 
        #x = torch.reshape(x, (-1,128))
        
        #unfold = nn.Unfold(kernel_size=(x.size(2), x.size(3)))
        #x = unfold(x).mean(1)  # Uporabite unfold za povprečno vrednost
        
        #x = self.cls_linear1(x)
        #x = torch.reshape(x, (-1,8,21))
        #x = self.cls_softmax(x)
        
        x = self.linear1(x)
        x = self.regression_head(x)
        
        return x

def train(model, loss_fn, optim, epochs, data_gen):
    loss_sum = 0
    eval_count = 0
    eval_sum = 0

    count = 1
    for img,corners,transformed_corners, homo_matrix,corner_offsets,orig_img in data_gen:
        
        #for i in range(len(img)):
        
            
        in_img = torch.moveaxis(img,-1,1)
        model.zero_grad()
        model_corner_offsets = model(in_img)
        #ones_tensor = torch.ones((len(img), 1))
        #model_homo_matrix = torch.cat((model_homo_matrix, ones_tensor), dim=1)
        #model_homo_matrix = torch.reshape(model_homo_matrix,(len(img),3,3))
        #corners_model = torch.reshape(corners_model,(-1,4,2))
        #corners_orig = corners[i]
        #model_homo_matrix, _ = cv2.findHomography(corners_orig.numpy(),corners_model.numpy())
        loss = loss_fn(corner_offsets,model_corner_offsets)
    
        loss.backward()
        optim.step()
        #visualize_flow(flow_out[0].detach().cpu().numpy())
        #visualize_flow(flow_gt[0].detach().cpu().numpy())
        loss_sum += loss.item()
        
        eval_sum += loss_sum
        eval_count += 1
        
        #for i in range(len(img)):
        #    show_corners_on_image(orig_img[i].detach().cpu().numpy(),corners[i].detach().cpu().numpy(),transformed_corners[i].detach().cpu().numpy(),corners[i].detach().cpu().numpy()+(torch.reshape(model_corner_offsets,(-1,4,2))[i].detach().cpu().numpy()*32-16))
        
        if count % 100 == 0:
            
            print(f"Epoch: {count}, Loss: {loss_sum/100}")
            torch.save(model.state_dict(), f"modelv1_{count}.pth")
            loss_sum = 0
        count += 1

    print("Total loss: ",eval_sum/eval_count)
        
model = HomoNet()
#model.load_state_dict(torch.load("modelv1_1100.pth"))
#model.eval()
optim = torch.optim.Adam(model.parameters(), lr=0.00001)
data_set = ImageDataSet("C:\\Users\\GTAbl\\Desktop\\RV\\Vaja1\\train2017")
loss_fn = nn.MSELoss()
data_gen = DataLoader(data_set, batch_size=64,shuffle=True)
train(model, loss_fn, optim, 90000, data_gen)