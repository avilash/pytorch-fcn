import torch
from torchvision import datasets, transforms
import numpy as np

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

input_transform = transforms.Compose([
	                   transforms.ToTensor(),
	                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	               ])

mask_transform = transforms.Compose([
	                   MaskToTensor(),
	               ])

restore_input_transform = transforms.Compose([
        transforms.Normalize((0.,0.,0.), (1.0/0.229, 1.0/0.224, 1.0/0.225)),
        transforms.Normalize((-0.485, -0.456, -0.406), (1., 1., 1.))
    ])
