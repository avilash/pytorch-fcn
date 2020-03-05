import os
import random
from config.base_config import cfg


class HERE(object):

    def __init__(self):
        pass

    def load(self, split_from_file=False):
        self.__imgs_dict = {}
        base_path = cfg.DATASETS.HERE.HOME
        img_path = os.path.join(base_path, 'Images')
        mask_path = os.path.join(base_path, 'Labels')

        train_file_names = []
        test_file_names = []

        if split_from_file:
            train_file_names = [l.strip('\n') for l in open(os.path.join(
                base_path, 'Splits', 'train.txt')).readlines()]
            test_file_names = [l.strip('\n') for l in open(os.path.join(
                base_path, 'Splits', 'test.txt')).readlines()]
        else:
            valid_images = [".jpg", ".png"]
            all_file_names = []
            for f in os.listdir(img_path):
                all_file_names.append(os.path.splitext(f)[0])

            num_files = len(all_file_names)
            num_test = num_files // 10

            random.shuffle(all_file_names)
            test_file_names = all_file_names[:num_test]
            train_file_names = all_file_names[num_test:]

        self.__train_imgs = []
        self.__train_labels = []
        self.__test_imgs = []
        self.__test_labels = []

        for file_name in train_file_names:
            self.__train_imgs.append(os.path.join(img_path, file_name + ".jpg"))
            self.__train_labels.append(os.path.join(mask_path, file_name + ".jpg"))

        for file_name in test_file_names:
            self.__test_imgs.append(os.path.join(img_path, file_name + ".jpg"))
            self.__test_labels.append(os.path.join(mask_path, file_name + ".jpg"))

        return self.__train_imgs, self.__train_labels, self.__test_imgs, self.__test_labels
