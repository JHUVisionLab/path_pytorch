from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
from torchvision import transforms, utils, models
from PIL import Image



import pdb


import nets 
class PathologyDataset(Dataset):
	"""Pathology dataset"""
	def __init__(self, img_dir, csv_file = 'microscopy_ground_truth.csv', transform=transforms.ToTensor(), shuffle = False, seed = 7):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied on a sample
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
		"""
		data = pd.read_csv(os.path.join(img_dir, "microscopy_ground_truth.csv"), header = None).values
		self.shuffle = shuffle
		#shuffle data
		if self.shuffle:
			for c in range(4):
				np.random.seed(seed)
				np.random.shuffle(data[0 + c*100:100 + c*100, :])



		img_ids = data[:,0]
		img_classes = data[:,1]
		img_labels = np.zeros((len(img_classes),), dtype=np.int32)
		class_to_label = {'Normal':0, 'Benign':1, "InSitu":2, 'Invasive':3}
		print('Class to label dictionary map: ')
		print(class_to_label)     

		for i in range(len(img_ids)):
			img_ids[i] = str(img_classes[i])+'/'+img_ids[i]
			#map from string class names to integer labels
			img_labels[i] = class_to_label[str(img_classes[i])]

		self.img_ids = img_ids
		self.img_labels = img_labels
		self.img_dir = img_dir
		self.transform = transform

	def __len__(self):
		return len(self.img_ids)

	def __getitem__(self, idx):
		img_name = os.path.join(self.img_dir,
								self.img_ids[idx])
		
		img = Image.open(img_name, mode='r')
		label = self.img_labels[idx]
		
		if self.transform:
			img = self.transform(img)

		return img, label
  

