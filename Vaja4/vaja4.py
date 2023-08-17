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
import csv


#get all files in folder and corresponding data from csv file
def get_images(dirpath):
    files = []

    for file in os.listdir(dirpath):
        if ".png" in file:
            tmp = []
            tmp.append(dirpath+file)
            files.append(dirpath+file)

    return files

def get_x_y_of_max(arr):
    max = 0
    x = 0
    y = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] > max:
                max = arr[i][j]
                x = i
                y = j
    return x,y

class DiceDataSet(Dataset):
    def __init__(self,path):
        self.path = path
        self.files = get_images(path)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        filename = self.files[index]
        image = cv2.imread(filename)
        width, height = image.shape[:2]
        dice = np.empty((0,4), dtype=np.int32)
        
        csvfile = open(self.path+'label_data.csv', 'r')
        reader = csv.reader(csvfile, delimiter=' ')
        
        color_dict = {
            "white": 0,
            "black": 1,
            "red": 2,
            "green": 3,
            "blue": 4,
            "yellow": 5,
            "orange": 6,
            "purple": 7,
            "pink": 8,
            "brown": 9,
            "grey": 10,
            "cyan": 11,
            "magenta": 12,
            "silver": 13,
            "gold": 14,
            "bronze": 15,
            "transparent": 16
        }
        
        for row in reader:
            if row[0] in filename:
                dice = np.append(dice,[[color_dict[row[1]],int(row[4]),int(row[3]),int(row[2])]],axis=0)
        
        cv2.imshow("img",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        vertical_move = random.randint(-16, 16)
        horizontal_move = random.randint(-16, 16)
        
        if vertical_move > 0:
            image = image[vertical_move:width,0:height]
        elif vertical_move < 0:
            image = image[0:width+vertical_move,0:height]
            
        width, height = image.shape[:2]

            
        if horizontal_move > 0:
            image = image[0:width,horizontal_move:height]
        elif horizontal_move < 0:
            image = image[0:width,0:height+horizontal_move]
            
        width, height = image.shape[:2]
            
        for d in dice:
            if vertical_move > 0:
                d[2] -= vertical_move
            if horizontal_move > 0:
                d[3] -= horizontal_move
            if d[2] < 0 or d[2] > width or d[3] < 0 or d[3] > height:
                d[2] = 0
                d[3] = 0
        
        
        if width < height:
            radius = width/2
        else:
            radius = height/2
        
        p = radius*sqrt(2)
        
        
        image_center = (int(height/2), int(width/2))
        rot_mat = cv2.getRotationMatrix2D(image_center, random.uniform(0.0,360.0), 1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        
        for d in dice:
            tmp_arr = np.zeros((width,height), dtype=np.uint8)
            tmp_arr[d[2],d[3]] = 255
            tmp_arr = cv2.warpAffine(tmp_arr, rot_mat, tmp_arr.shape[1::-1])

            if np.sum(tmp_arr) == 0:
                d[2] = 0
                d[3] = 0
            else:
                d[2] = np.where(tmp_arr != 0)[0][0]
                d[3] = np.where(tmp_arr != 0)[1][0]
                
        width, height = image.shape[:2]

                
        scale = random.uniform(0.8, 1.5)
        image = cv2.resize(image, (int(height*scale), int(width*scale)))
        p *= scale
                
        for d in dice:
            tmp_arr = np.zeros((width,height), dtype=np.uint8)
            tmp_arr[d[2],d[3]] = 255

            tmp_arr = cv2.resize(tmp_arr, (int(height*scale), int(width*scale)), interpolation=cv2.INTER_CUBIC)

            if np.sum(tmp_arr) == 0:
                d[2] = 0
                d[3] = 0
            else:
                d[2] = np.where(tmp_arr != 0)[0][0]
                d[3] = np.where(tmp_arr != 0)[1][0]    
        
        width, height = image.shape[:2]
        
        
        
        
        image = image[int(width/2-p/2):int(width/2+p/2),int(height/2-p/2):int(height/2+p/2)]
        for d in dice:
            if d[2] < int(width/2-p/2) or d[2] > int(width/2+p/2) or d[3] < int(height/2-p/2) or d[3] > int(height/2+p/2):
                d[2] = 0
                d[3] = 0
            else:
                d[2] -= int(width/2-p/2)
                d[3] -= int(height/2-p/2)
        
        
        for d in dice:
            if d[2] != 0 and d[3] != 0:
                image = cv2.circle(image, (d[3],d[2]), int(16*scale), (0,0,255), 2)
        
        cv2.imshow("img",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return image, dice
    
    
dicedataset = DiceDataSet("C:\\Users\\GTAbl\\Desktop\\RV\\Vaja4\\data\\data_2018_09_11\\")

diceloader = DataLoader(dicedataset, batch_size=1, shuffle=True)

for img,dice in diceloader:
    print(img.shape)
    print(dice)
    break
