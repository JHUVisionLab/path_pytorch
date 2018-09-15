from __future__ import print_function, division
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
from torch.utils.data import random_split
from torchvision import transforms, utils, models
from PIL import Image
from cross_validation import k_folds

import pandas as pd


import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # useful stateless functions

import nets 
from PathologyDataset import PathologyDataset
#### Settings 

USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 4

print('using device:', device)


def check_accuracy(loader, model, train, filename=None):
	"""evaluted model and report accuracy

	args:
		loader: pytorch dataloader
		model: pytorch module
		train (boolean): in training mode or not

	return:
		acc: accuracy of evaluation 
	"""
	num_correct = 0
	num_samples = 0
	model.eval()  # set model to evaluation mode
	
	if train:
		print('Checking accuracy on validation set')
	else:
		print('Final Evaluation') 

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)
			scores = model(x)
			_, preds = scores.max(1)
			num_correct += (preds == y).sum()
			num_samples += preds.size(0)
			if not train:
				scores = nn.Softmax(scores).data.cpu().numpy()
				c0, c1, c2, c3 = np.split(scores, 4, axis = 1)
				y = y.data.cpu().numpy()
				preds = preds.data.cpu().numpy()
				results_dict = {'p0': c0, 'p1': c1, 'p2': c2, 'p3': c3, 'label': y, 'pred': preds, 'eval': preds == y}
				results = pd.DataFrame.from_dict(results_dict)
				results.to_csv(filename)

		
		acc = float(num_correct) / num_samples
		print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
		return acc


def train_loop(model, loaders, optimizer, epochs=10, filename=None):
	"""
	Train a model on CIFAR-10 using the PyTorch Module API.
	
	Inputs:
	- model: A PyTorch Module giving the model to train.
	- optimizer: An Optimizer object we will use to train the model
	- epochs: (Optional) A Python integer giving the number of epochs to train for
	
	Returns: Nothing, but prints model accuracies during training.
	"""
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
	
	model = model.to(device=device)  # move the model parameters to CPU/GPU
	loader_train = loaders['train']
	loader_val = loaders['val']
	print('training begins')
	for e in range(epochs):
		for t, (x, y) in enumerate(loader_train):
			model.train()  # put model to training mode

			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)

			scores = model(x)
			loss = F.cross_entropy(scores, y)

			# Zero out all of the gradients for the variables which the optimizer
			# will update.
			optimizer.zero_grad()

			# This is the backwards pass: compute the gradient of the loss with
			# respect to each  parameter of the model.
			loss.backward()

			# Actually update the parameters of the model using the gradients
			# computed by the backwards pass.
			optimizer.step()
			if t % print_every == 0 :
				print('Epoch %d of %d, Iteration %d, loss = %.4f' % (e+1, epochs, t+1, loss.item()))
				acc = check_accuracy(loader_val, model, train=True)
				print()

	acc = check_accuracy(loader_val, model, train=False, filename=filename)
	return acc

def train_network(ssh = True):
	NUM_TRAIN = 360
	NUM_VAL = 40
	batch_size = 50
	learning_rate = 1e-2
	k = 10
	num_classes = 4
	transformation_train = transforms.Compose([transforms.RandomChoice([transforms.Resize([224, 224]), 
																		transforms.RandomCrop([224, 224]),
																		transforms.RandomResizedCrop(224)]),
											   transforms.RandomApply([transforms.ColorJitter()]),
											   transforms.RandomVerticalFlip(),
											   transforms.RandomHorizontalFlip(),
											   transforms.ToTensor(),
											   transforms.Normalize(mean=[0.485, 0.456, 0.406],
															std=[0.229, 0.224, 0.225])
											   ])

	transformation_val = transforms.Compose([transforms.Resize([224, 224]),
											 transforms.ToTensor(),
											 transforms.Normalize(mean=[0.485, 0.456, 0.406],
															std=[0.229, 0.224, 0.225])
											])
	if ssh:
		root_dir='/workspace/path_data/Part-A_Original'
	else:
		root_dir='/Users/admin/desktop/path_pytorch/Part-A_Original'

	# path_data_train and path_data_val should have different transformation (path_data_val should not apply data augmentation)
	# therefore we shuffle path_data_train and copy its shuffled image ids and corresponding labels over to path_data_val
	path_data_train = PathologyDataset(csv_file='microscopy_ground_truth.csv', root_dir=root_dir, shuffle = True, transform=transformation_train)
	path_data_val = PathologyDataset(csv_file='microscopy_ground_truth.csv', root_dir=root_dir, shuffle = False, transform=transformation_val)

	path_data_val.img_ids = path_data_train.img_ids.copy()
	path_data_val.img_labels = path_data_train.img_labels.copy()

	# make sure the two datasets are identical in ids and labels
	assert np.all(np.equal(path_data_val.img_ids, path_data_train.img_ids))
	assert np.all(np.equal(path_data_val.img_labels, path_data_train.img_labels))

	acc = np.zeros((k,))
	counter = 0
	for train_idx, test_idx in k_folds(n_splits = k):
		filename = 'results_' + str(k) + '.txt'
		
		loader_train = torch.utils.data.DataLoader(dataset = path_data_train, batch_size = batch_size, sampler = sampler.SubsetRandomSampler(train_idx))
		loader_val = torch.utils.data.DataLoader(dataset = path_data_val, batch_size = 40, sampler = sampler.SubsetRandomSampler(test_idx))
	
		model = nets.resnet50(num_classes)
		optimizer = optim.RMSprop(model.parameters())
		loaders = {'train': loader_train, 'val': loader_val}
		acc[counter] = train_loop(model, loaders, optimizer, epochs=1, filename=filename)

		counter+=1
	
	print('k-fold CV accuracy: ', acc)
	print('final mean accuracy: ', np.mean(acc))

train_network()
