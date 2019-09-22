import _init_paths
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from models.fcn import Resnet
from models.fcn2 import Resnet2
from dataloader import gor, here
from img_loader import GORLoader, HERELoader
from transforms import input_transform, restore_input_transform, mask_transform

from gen_utils import make_dir_if_not_exist

from base_config import cfg, cfg_from_file

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
	torch.manual_seed(1)
	if args.cuda:
		torch.cuda.manual_seed(1)

	exp_dir = os.path.join("data", args.exp_name)
	make_dir_if_not_exist(exp_dir)

	model = Resnet(args.num_classes)
	model = nn.DataParallel(model, device_ids=args.gpu_devices)
	model.to(device)

	model_dict = None
	if args.ckp:
		if os.path.isfile(args.ckp):
			print("=> Loading checkpoint '{}'".format(args.ckp))
			model_dict = torch.load(args.ckp)
			print("=> Loaded checkpoint '{}'".format(args.ckp))

	if (args.mode == 'demo') and (model_dict is None):
		print ("Please specify model path")
		return

	if model_dict is not None:
		model.load_state_dict(model_dict['state_dict'])

	cudnn.benchmark = True

	params = []
	for key, value in dict(model.named_parameters()).items():
		if value.requires_grad:
			params += [{'params':[value]}]

	criterion = torch.nn.NLLLoss(None, ignore_index=255)
	optimizer = optim.Adam(params, lr=args.lr)

	if args.mode == 'demo':
		train_data_loader, test_data_loader = sample_data(args.dset)
		test(test_data_loader, model, criterion, demo=True)
		return

	for epoch in range(1, args.epochs + 1):
		train_data_loader, test_data_loader = sample_data(args.dset)
		test(test_data_loader, model, criterion)
		train(train_data_loader, model, criterion, optimizer, epoch)
		model_to_save = {
			"epoch" : epoch + 1,
			'state_dict': model.state_dict(),
		}
		if epoch%args.ckp_freq == 0:
			file_name = os.path.join(exp_dir, "checkpoint_" + str(epoch) + ".pth")
			save_checkpoint(model_to_save, file_name)

def sample_data(dset):
	train_data_loader = None
	test_data_loader = None
	
	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

	train_data_loader = None
	test_data_loader = None

	if dset == 'here':
		dataset = here.HERE()
		train_imgs, train_labels, test_imgs, test_labels = dataset.load(split_from_file=True)

		train_data_loader = torch.utils.data.DataLoader(
			HERELoader(train_imgs, train_labels, img_transform=input_transform, label_transform=mask_transform),
			batch_size=args.batch_size, shuffle=True, **kwargs)
		test_data_loader = torch.utils.data.DataLoader(
			HERELoader(test_imgs, test_labels, img_transform=input_transform, label_transform=mask_transform),
			batch_size=args.batch_size, shuffle=True, **kwargs)

	if dset == 'gor':
		dataset = gor.GOR()
		train_imgs, train_labels, test_imgs, test_labels = dataset.load()

		train_data_loader = torch.utils.data.DataLoader(
			GORLoader(train_imgs, train_labels, img_transform=input_transform, label_transform=mask_transform),
			batch_size=args.batch_size, shuffle=True, **kwargs)
		test_data_loader = torch.utils.data.DataLoader(
			GORLoader(test_imgs, test_labels, img_transform=input_transform, label_transform=mask_transform),
			batch_size=args.batch_size, shuffle=True, **kwargs)

	return train_data_loader, test_data_loader

def train(data_loader, model, criterion, optimizer, epoch):
	print ("******** Training ********")
	total_loss = 0
	model.train()
	for batch_idx, data in enumerate(data_loader):
		batch_idx += 1
		imgs, labels = data		
		imgs, labels = imgs.to(device), labels.to(device)
		imgs, labels = Variable(imgs), Variable(labels)

		outputs = model(imgs)

		loss = criterion(outputs, labels)
		total_loss += loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		log_step = args.train_log_step
		if batch_idx%log_step == 0:
			print('Train Epoch: {} [{}/{}] \t Loss: {:.4f}'.format(epoch, batch_idx, len(data_loader), total_loss/log_step))
			total_loss = 0
	print ("****************")

def test(data_loader, model, criterion, demo=False):
	print ("******** Testing ********")
	if demo:
		img_target_dir = os.path.join('data', args.exp_name, 'results', 'imgs')
		weighted_img_target_dir = os.path.join('data', args.exp_name, 'results', 'wt_imgs')
		pred_target_dir = os.path.join('data', args.exp_name, 'results', 'preds')
		make_dir_if_not_exist(img_target_dir)
		make_dir_if_not_exist(weighted_img_target_dir)
		make_dir_if_not_exist(pred_target_dir)
	with torch.no_grad():
		model.eval()
		total_loss = 0
		for batch_idx, data in enumerate(data_loader):
			batch_idx += 1
			imgs, labels = data		
			imgs, labels = imgs.to(device), labels.to(device)
			imgs, labels = Variable(imgs), Variable(labels)

			outputs = model(imgs)

			loss = criterion(outputs, labels)
			total_loss += loss

			if demo:
				predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
				for idx, img, prediction in zip(range(args.batch_size), imgs, predictions):
					real_img = restore_input_transform(img.data.cpu())
					real_img = real_img.numpy()*255
					real_img = real_img.astype(np.uint8)
					real_img = np.moveaxis(real_img, 0, -1)
					prediction[prediction == 1] = 255

					red_mask = np.zeros(real_img.shape, real_img.dtype)
					red_mask[prediction == 255] = (0, 0, 255)
					weighted_img = cv2.addWeighted(red_mask, 0.4, real_img, 1, 0)

					cv2.imwrite(os.path.join(img_target_dir, str(batch_idx) + "_" + str(idx) + ".jpg"), real_img)
					cv2.imwrite(os.path.join(weighted_img_target_dir, str(batch_idx) + "_" + str(idx) + ".jpg"), weighted_img)
					cv2.imwrite(os.path.join(pred_target_dir, str(batch_idx) + "_" + str(idx) + "_pred.png"), prediction)

		print('Test Loss: {}'.format(total_loss/len(data_loader)))
	print ("****************")

def save_checkpoint(state, file_name):
    torch.save(state, file_name)

if __name__ == '__main__':	
	parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
	parser.add_argument('--mode', default='train', type=str,
	                help='Mode - Train-Test / Demo')
	parser.add_argument('--exp_name', default='exp0', type=str,
	                help='name of experiment')
	parser.add_argument('--cuda', action='store_true', default=False,
	                help='enables CUDA training')
	parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, 
					help="List of GPU Devices to train on")
	parser.add_argument('--epochs', type=int, default=15, metavar='N',
	                help='number of epochs to train (default: 10)')
	parser.add_argument('--ckp_freq', type=int, default=1, metavar='N',
	                help='Checkpoint Frequency (default: 1)')
	parser.add_argument('--batch_size', type=int, default=4, metavar='N',
	                help='input batch size for training (default: 64)')
	parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
	                help='learning rate (default: 0.0001)')
	parser.add_argument('--ckp', default=None, type=str,
	                help='path to load checkpoint')
	
	parser.add_argument('--train_log_step', type=int, default=10, metavar='M',
	                help='Number of iterations after which to log the loss')

	parser.add_argument('--num_classes', type=int, default=2, metavar='NC',
	                help='Num Classes (default: 2)')

	parser.add_argument('--dset', type=str, default='here',
	                help='Dataset')


	global args, device
	args = parser.parse_args()
	args.cuda = args.cuda and torch.cuda.is_available()
	cfg_from_file("config/test.yaml")

	if args.cuda:
		device = 'cuda'
		if args.gpu_devices is None:
			args.gpu_devices = [0]
	else:
		device = 'cpu'
	main()
