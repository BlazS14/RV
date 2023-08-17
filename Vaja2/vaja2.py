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
    color = img[0,0]
    return img
def generate_rectangle(img):
    height, width = img.shape[:2]
    
    if height <width:
        max = height-40
    else:
        max = width-40
    
    # generate an image and a random rectangle
    points = generate_convex_quadrilateral_points(0,max)
    cv2.drawContours(img, [np.array(points)], 0, (random.randint(0, 256),random.randint(0, 256),random.randint(0, 256)), -1)
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

def appy_homography_to_image(img,points):
    height, width = img.shape[:2]
    
    hbuffer = int(height/4)
    wbuffer = int(width/4)
    
    p1 = [random.randint(0, hbuffer), random.randint(0, wbuffer)]
    p2 = [random.randint(0, hbuffer), random.randint(width-wbuffer, width)]
    p3 = [random.randint(height-hbuffer, height), random.randint(width-wbuffer, width)]
    p4 = [random.randint(height-hbuffer, height), random.randint(0, wbuffer)]
    
    plist = [p1,p2,p3,p4]
    
    for i in range(random.randint(0,3)):
        tmp = plist.pop(0)
        plist.append(tmp)
    
    homography_matrix = cv2.getPerspectiveTransform(np.float32(plist),np.float32([[0,0],[width,0],[width,height],[0,height]]))
    
    img = cv2.warpPerspective(img, homography_matrix, (width,height))
    
    #apply homography to points
    i = 0
    l = len(points)
    while i < l:
        p = np.array([points[i][0],points[i][1],1])
        p = np.matmul(homography_matrix,p)
        p = np.array([p[0]/p[2],p[1]/p[2]])
        #check if point is NOT inside image
        if p[0] < 0 or p[0] > width or p[1] < 0 or p[1] > height:
            np.delete(points,i)
            l -= 1
        else:
            i += 1
    
    return img, points
    
#check if all rgb colors in image differ by a certain amount    
def get_color_differnece(img,amount):

    #get all unique rgb values in image into a list
    unique_colors = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for u in range(len(unique_colors)):
                if np.array_equal(img[i,j],unique_colors[u]):
                    break
                unique_colors.append(img[i,j])
                
    #check if all rgb values differ by a certain amount
    for i in range(len(unique_colors)):
        for j in range(len(unique_colors)):
            if not (abs(unique_colors[i][0]-unique_colors[j][0]) > amount or abs(unique_colors[i][1]-unique_colors[j][1]) > amount or abs(unique_colors[i][2]-unique_colors[j][2]) > amount):
                return False
    return True


def generate_image(shepe_type,height,width):
    while True:
        img = create_image(height,width)
        if shepe_type == 0:
            img, points = generate_rectangle(img)
        elif shepe_type == 1:
            img, points = generate_triangle(img)
        elif shepe_type == 2:
            img, points = generate_star(img)
        
        img,points = appy_homography_to_image(img,points)
        
        if get_color_differnece(img,50):
            if len(points) > 0:
                break
    return img, points
    
    
    
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
        self.resnet1 = ResNet(3, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.resnet2 = ResNet(64, 64)
        self.resnet3 = ResNet(64, 128)
        self.resnet4 = ResNet(128, 128)
        self.linear1 = nn.Linear(8192,512)
        
        self.conv1 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(256, 65, kernel_size=1, stride=1, padding=1, bias=False)
        
        
        self.softmax = nn.Softmax(dim=2)
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
       
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.softmax(x)


        return x
    
    
    
    
    
    
    
    
    
while True:
    img, points = generate_image(random.randint(0,2),320,320)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()