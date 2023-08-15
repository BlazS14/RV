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
#torch.set_default_device(dml)



#read .flo file and convert it to a numpy array
def read_flo_file(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            return np.resize(data, (int(h), int(w), 2))
        
#visualize .flo data from numpy array
def visualize_flow(flow):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    #get the range of the flow values
    max_val = np.max(flow)
    min_val = np.min(flow)
    #get the flow values in the range of 0-1
    u_norm = (u - min_val) / (max_val - min_val)
    v_norm = (v - min_val) / (max_val - min_val)
    #get the flow values in the range of 0-255
    u_norm = u_norm * 255
    v_norm = v_norm * 255
    #create a flow image
    flow_img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_img[:, :, 0] = u_norm
    flow_img[:, :, 1] = v_norm
    #convert the image to RGB
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)
    #show the image
    plt.imshow(flow_img)
    plt.show()
    
def save_flow(flow,name):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    #get the range of the flow values
    max_val = np.max(flow)
    min_val = np.min(flow)
    #get the flow values in the range of 0-1
    u_norm = (u - min_val) / (max_val - min_val)
    v_norm = (v - min_val) / (max_val - min_val)
    #get the flow values in the range of 0-255
    u_norm = u_norm * 255
    v_norm = v_norm * 255
    #create a flow image
    flow_img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_img[:, :, 0] = u_norm
    flow_img[:, :, 1] = v_norm
    #convert the image to RGB
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)
    #show the image
    cv2.imwrite(name, flow_img)

#get all files in folder and its subfolder and return an array of full paths to these files
def get_frames_in_folder(path):
    files = np.empty((0,2))
    for r, d, f in os.walk(path):
        prevfile = ""
        for file in f:
            if prevfile != "":
                files = np.append(files, [[prevfile, os.path.join(r, file)]], axis=0)
            prevfile = os.path.join(r, file)
    return files


def get_flows_in_folder(path):
    files = np.empty((0,1))
    for r, d, f in os.walk(path):
        for file in f:
            files = np.append(files, [[os.path.join(r, file)]], axis=0)
    return files

#def a function that crops the first two dimensions of a numpy array to be devisable by 8
def crop_array(flow):
    h = flow.shape[0]
    w = flow.shape[1]
    h = h - h%8
    w = w - w%8
    return flow[0:h, 0:w, :]

class FlowDataSet(Dataset):
    def __init__(self,path):
        self.flowarray = get_flows_in_folder(path+"/flow")
        self.framearray = get_frames_in_folder(path+"/final")
        
    def __len__(self):
        return len(self.flowarray)
    
    def __getitem__(self, index):
        flow = read_flo_file(self.flowarray[index][0])
        #read image to variable
        frame1 = cv2.imread(self.framearray[index][0])
        frame2 = cv2.imread(self.framearray[index][1])
        frame1 = frame1.astype(np.float32)
        frame2 = frame2.astype(np.float32)
        frame1 = frame1/255
        frame2 = frame2/255
        frame1 = crop_array(frame1)
        frame2 = crop_array(frame2)
        flow = crop_array(flow)
        #resize both frames and flow to be devisable by 8
        
        
        return flow, frame1, frame2
    
#define a FlowNetSimple down block
'''class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        return x'''
    
class MaxPoolBlock(nn.Module):
    def __init__(self):
        super(MaxPoolBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
        
    def forward(self, x):
        x = self.maxpool(x)
        return x
    
    
class ConvBatchReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBatchReLUBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.convtranspose1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) #padding?
        
    def forward(self, x):
        x = self.convtranspose1(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv2d(x)
        return x
    
class ConcatBlock(nn.Module):
    def __init__(self):
        super(ConcatBlock, self).__init__()
        
    def forward(self, x,y):
        x = torch.cat((x,y), dim=1)
        return x
    
#define a FlownetSimple encoder
'''class FlowNetSimpleEncoder(nn.Module):
    def __init__(self):
        super(FlowNetSimpleEncoder, self).__init__()
        self.down1 = FlowNetSimpleDownBlock(6, 32)
        self.down2 = FlowNetSimpleDownBlock(32, 64)
        self.down3 = FlowNetSimpleDownBlock(64, 128)

        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        return x
    
#define a fllownetSimple decoder
class FlowNetSimpleDecoder(nn.Module):
    def __init__(self):
        super(FlowNetSimpleDecoder, self).__init__()
        self.convtranspose1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convtranspose2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.convtranspose3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        
        
    def forward(self, x):
        x = self.convtranspose1(x)
        x = self.conv1(x)
        x = self.convtranspose2(x)
        x = self.conv2(x)
        x = self.convtranspose3(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x'''


#define a FlowNetSimple model
class FlowNetSimple(nn.Module):
    def __init__(self):
        super(FlowNetSimple, self).__init__()
        self.cbr1 = ConvBatchReLUBlock(6, 32)
        self.cbr2 = ConvBatchReLUBlock(32, 64)
        self.cbr3 = ConvBatchReLUBlock(64, 128)
        self.cbr4 = ConvBatchReLUBlock(128, 256)

        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(256, 64)
        self.up3 = UpBlock(128, 32)
        self.up2x2 = UpBlock(2, 2)
        
        self.concat = ConcatBlock()
        self.maxpool = MaxPoolBlock()
        
        self.conv1 = ConvBlock(256, 2)
        self.conv2 = ConvBlock(258, 2)
        self.conv3 = ConvBlock(130, 2)
        
        self.conv4 = ConvBlock(66, 2)
        
        
        
    def forward(self, x):
        l11 = self.cbr1(x)
        l21 = self.maxpool(l11)
        l21 = self.cbr2(l21)
        l31 = self.cbr3(self.maxpool(l21))
        l41 = self.cbr4(self.maxpool(l31))
        
        l32 = self.concat(self.up1(l41),l31)
        l22 = self.concat(self.up2(l32), l21)
        l12 = self.concat(self.up3(l22), l11)
        
        l42 = self.conv1(l41)
        
        l33 = self.concat(self.up2x2(l42), l32)
        l34 = self.conv2(l33)
        
        l23 = self.concat(self.up2x2(l34), l22)
        l24 = self.conv3(l23)
        
        l13 = self.concat(self.up2x2(l24), l12)
        l14 = self.conv4(l13)
        
        return l14
    

class EPELoss(nn.Module):
    def __init__(self):
        super(EPELoss,self).__init__()

    def forward(self,flow_gt, flow):
        error = torch.sqrt(torch.sum((flow_gt - flow) ** 2, axis=-1))
        epe = torch.mean(error)
        return epe
                

def train(model, loss_fn, optimizer, epochs, dataset):
    loss_sum = 0
    loss_sum_cv = 0
    for epoch in range(1,epochs+1):
        
        for flow_gt,f1,f2 in dataset:
            f1 = torch.moveaxis(f1, -1, 1)
            f2 = torch.moveaxis(f2, -1, 1)
            model_input = torch.cat((f1,f2), dim=1)
            #model.zero_grad()
            flow_out = model(model_input)
            flow_out = torch.moveaxis(flow_out, 1, -1)
            loss = loss_fn(flow_gt,flow_out)
            
            flow_out_cv = None
            
            f1 = cv2.cvtColor(torch.moveaxis(f1, 1, -1)[0].detach().cpu().numpy(), cv2.COLOR_BGR2GRAY)
            f2 = cv2.cvtColor(torch.moveaxis(f2, 1, -1)[0].detach().cpu().numpy(), cv2.COLOR_BGR2GRAY)
            flow_out_cv = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #loss.backward()
            loss_cv = loss_fn(flow_gt,flow_out_cv)

            #optimizer.step()
            #visualize_flow(flow_out[0].detach().cpu().numpy())
            #visualize_flow(flow_gt[0].detach().cpu().numpy())
            loss_sum += loss.item()
            loss_sum_cv += loss_cv.item()
            
            
        if epoch % 1 == 0:
            print(f"Epoch: {epoch}, Loss: {loss_sum/1}, Loss_cv: {loss_sum_cv/1}")
            loss_sum = 0
            loss_sum_cv = 0
            #torch.save(model.state_dict(), f"model_{epoch}.pth")
            
            
model = FlowNetSimple()
model.load_state_dict(torch.load("model_480.pth"))
torchinfo.summary(model, (1, 6, 384, 512))
model.eval()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
data_set = FlowDataSet("C:\\Users\\GTAbl\\Desktop\\RV\\Vaja3\\data\\test")
loss_fn = EPELoss()
data_gen = DataLoader(data_set, batch_size=8,shuffle=True)
#train(model, loss_fn, optim, 500000, data_gen)
prev = None
counter = 0
for p in get_frames_in_folder("C:\\Users\\GTAbl\\Desktop\\RV\\Vaja3\\data\\poki"):
    if prev != None:
        f1 = cv2.imread(prev)
        f2 = cv2.imread(p[1])
        f1 = f1.astype(np.float32)
        f2 = f2.astype(np.float32)
        f1 = f1/255
        f2 = f2/255
        f1 = crop_array(f1)
        f2 = crop_array(f2)
        f1 = torch.from_numpy(f1)
        f2 = torch.from_numpy(f2)
        f1 = torch.moveaxis(f1, -1, 0)
        f2 = torch.moveaxis(f2, -1, 0)
        model_input = torch.cat((f1,f2), dim=0)
        model_input = model_input.unsqueeze(0)
        flow_out = model(model_input)
        flow_out = torch.moveaxis(flow_out, 1, -1)
        save_flow(flow_out[0].detach().cpu().numpy(), f"C:\\Users\\GTAbl\\Desktop\\RV\\Vaja3\\data\\poki\\flow_{counter}.png")
        counter += 1
    prev = p[1]