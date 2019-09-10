import os
import os.path

import cv2
import numpy as np

import torch.utils.data
import torchvision.transforms as transforms

class VOCLoader(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, img_transform=None, label_transform=None):
        self.imgs = imgs
        self.labels = labels
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        img_pth = self.imgs[index]
        label_pth = self.labels[index]

        img = cv2.imread(img_pth)
        # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        img = cv2.resize(img, (640, 360))
        label = cv2.imread(label_pth, cv2.IMREAD_GRAYSCALE)
        try:
            # label = cv2.resize(label, (0,0), fx=0.5, fy=0.5)
            label = cv2.resize(label, (640, 360))
        except Exception as e:
            label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)        

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)
        
        return img, label

    def __len__(self):
        return len(self.imgs)