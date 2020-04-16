import os
import random
import re
from config.base_config import cfg


class OpenEDS(object):

    def __init__(self):
        self.dsets = []

    def load(self):
        self.__imgs_dict = {}
        base_path = cfg.DATASETS.OPENEDS.HOME
        self.dsets = []
        folders = ["train", "validation", "test"]
        for folder in folders:
            images_path = os.path.join(base_path, folder, "images")
            labels_path = os.path.join(base_path, folder, "labels")
            images = []
            labels = []

            files = sorted([f for f in os.listdir(images_path) if re.match(r'.*\.png', f)])
            file_names = []
            for f in files:
                file_names.append(os.path.splitext(f)[0])
            random.shuffle(file_names)
            for f in file_names:
                images.append(os.path.join(images_path, f + ".png"))
                labels.append(os.path.join(labels_path, f + ".npy"))

            self.dsets.append([images, labels])

        return self.dsets[0][0], self.dsets[0][1], self.dsets[1][0], self.dsets[1][1], self.dsets[2][0], self.dsets[2][1]
