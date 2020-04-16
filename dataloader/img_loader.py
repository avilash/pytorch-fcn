import cv2
import numpy as np

import torch.utils.data


class BaseLoader(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, img_transform=None, label_transform=None):
        self.imgs = imgs
        self.labels = labels
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        img_pth = self.imgs[index]
        label_pth = self.labels[index]
        default_height = 480
        default_width = 640

        img = cv2.imread(img_pth)
        try:
            img = cv2.resize(img, (default_width, default_height))
        except Exception as e:
            img = np.zeros((default_height, default_width, 3), dtype=np.uint8)

        label = cv2.imread(label_pth, cv2.IMREAD_GRAYSCALE)
        try:
            label = cv2.resize(label, (default_width, default_height))
        except Exception as e:
            label = np.zeros((default_height, default_width), dtype=np.uint8)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def __len__(self):
        return len(self.imgs)


class GORLoader(BaseLoader):
    def __init__(self, imgs, labels, img_transform=None, label_transform=None):
        super(GORLoader, self).__init__(imgs, labels, img_transform=img_transform, label_transform=label_transform)

    def __getitem__(self, index):
        img, label = super(GORLoader, self).__getitem__(index)
        return img, label


class HERELoader(BaseLoader):
    def __init__(self, imgs, labels, img_transform=None, label_transform=None):
        super(HERELoader, self).__init__(imgs, labels, img_transform=img_transform, label_transform=label_transform)

    def __getitem__(self, index):
        img, label = super(HERELoader, self).__getitem__(index)
        label[label != 0] = 1
        return img, label


class OPENEDSLoader(BaseLoader):
    def __init__(self, imgs, labels, img_transform=None, label_transform=None):
        super(OPENEDSLoader, self).__init__(imgs, labels, img_transform=img_transform, label_transform=label_transform)

    def __getitem__(self, index):
        img_pth = self.imgs[index]
        label_pth = self.labels[index]
        default_height = int(640/1)
        default_width = int(400/1)

        img = cv2.imread(img_pth)
        try:
            img = cv2.imread(img_pth)
            img = cv2.resize(img, (default_width, default_height))
        except Exception as e:
            img = np.zeros((default_height, default_width, 3), dtype=np.uint8)

        try:
            label = np.load(label_pth)
            label = cv2.resize(label, (default_width, default_height))
        except Exception as e:
            label = np.zeros((default_height, default_width), dtype=np.uint8)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label
