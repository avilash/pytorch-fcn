import os
import random
from base_config import cfg

import cv2

class GOR(object):

    def __init__(self):
    	pass

    def load(self):
    	self.__imgs_dict = {}
    	base_path = cfg.DATASETS.GOR.HOME
    	img_path = os.path.join(base_path, 'img')
        mask_path = os.path.join(base_path, 'cls_mask')

        train_file_names = [l.strip('\n') for l in open(os.path.join(
            base_path, 'ImageSets', 'Main', 'train.txt')).readlines()]
        test_file_names = [l.strip('\n') for l in open(os.path.join(
            base_path, 'ImageSets', 'Main', 'test.txt')).readlines()]

        self.__train_imgs = []
        self.__train_labels = []
        self.__test_imgs = []
        self.__test_labels = []

        for file_name in train_file_names:
        	self.__train_imgs.append(os.path.join(img_path, file_name+".jpg"))
        	self.__train_labels.append(os.path.join(mask_path, file_name+".png"))

        for file_name in test_file_names:
        	self.__test_imgs.append(os.path.join(img_path, file_name+".jpg"))
        	self.__test_labels.append(os.path.join(mask_path, file_name+".png"))

        return self.__train_imgs, self.__train_labels, self.__test_imgs, self.__test_labels