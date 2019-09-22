import _init_paths
import os
import random

from gen_utils import make_dir_if_not_exist

from base_config import cfg, cfg_from_file

def writeListToFileDetect(listname, filename):
    f = open(filename, 'w')

    if(len(listname)>0):
        f.write(listname[0])
    for data in listname[1:]:
        f.write("\n" + data)  # python will convert \n to os.linesep
    f.close()

def split_here_train_test(base_path, test_percent):
	img_path = os.path.join(base_path, 'Images')
	mask_path = os.path.join(base_path, 'Labels')

	valid_images = [".jpg",".png"]
	all_file_names = []
	for f in os.listdir(img_path):
		all_file_names.append(os.path.splitext(f)[0])

	num_files = len(all_file_names)
	num_test = num_files//test_percent

	random.shuffle(all_file_names)
	test_file_names = all_file_names[:num_test]
	train_file_names = all_file_names[num_test:]

	dst_path = os.path.join(base_path, "Splits")
	make_dir_if_not_exist(dst_path)

	writeListToFileDetect (train_file_names, os.path.join(dst_path, 'train.txt'))
	writeListToFileDetect (test_file_names, os.path.join(dst_path, 'test.txt'))

if __name__ == "__main__":
	cfg_from_file("config/test.yaml")
	split_here_train_test(cfg.DATASETS.HERE.HOME, 10)