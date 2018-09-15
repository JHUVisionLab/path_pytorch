from __future__ import print_function, division
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
from torch.utils.data import random_split
from torchvision import transforms, utils, models
from PIL import Image


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
print_every = 10

print('using device:', device)


def check_accuracy(loader, model, train):
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
		print('Checking accuracy on test set')   

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)
			scores = model(x)
			_, preds = scores.max(1)
			num_correct += (preds == y).sum()
			num_samples += preds.size(0)
		acc = float(num_correct) / num_samples
		print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
		return acc


def train_loop(model, loaders, optimizer, epochs=10):
	"""
	Train a model on CIFAR-10 using the PyTorch Module API.
	
	Inputs:
	- model: A PyTorch Module giving the model to train.
	- optimizer: An Optimizer object we will use to train the model
	- epochs: (Optional) A Python integer giving the number of epochs to train for
	
	Returns: Nothing, but prints model accuracies during training.
	"""
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
				print('Epoch %d, Iteration %d, loss = %.4f' % (e+1, t+1, loss.item()))
				acc = check_accuracy(loader_val, model, train=True)
				print()


	return check_accuracy(loader_val, model, train=True)

def train_network(ssh = True):
	NUM_TRAIN = 360
	NUM_VAL = 40
	batch_size = 8
	learning_rate = 1e-2
	transformation = transforms.Compose([transforms.Resize([224, 224]),
									 transforms.RandomVerticalFlip(),
									 transforms.RandomHorizontalFlip(),
									 transforms.ToTensor(),
									 transforms.Normalize(mean=[0.485, 0.456, 0.406],
														  std=[0.229, 0.224, 0.225])
									])
	if ssh:
		root_dir='/workspace/path_data/Part-A_Original'
	else:
		root_dir='/Users/admin/desktop/path_pytorch/Part-A_Original'

	
	path_data = PathologyDataset(csv_file='microscopy_ground_truth.csv', root_dir=root_dir, transform=transformation)
	model = nets.resnet50(4)
	#model = nets.TwoLayerFC(input_size=224, hidden_size=512, num_classes=4)
	optimizer = optim.rmsprop(model.parameters())
	path_data_train, path_data_val = random_split(path_data,[NUM_TRAIN, NUM_VAL])

	loader_train = DataLoader(path_data_train,batch_size=batch_size, shuffle = True)

	loader_val = DataLoader(path_data_val, batch_size=batch_size, shuffle = True)
	loaders = {'train': loader_train, 'val': loader_val}
	acc = train_loop(model, loaders, optimizer, epochs=20)
	print('final accuracy: ', acc)

	for param in model.parameters():
      param.requires_grad = True
    optimizer = optim.rmsprop(model.parameters(), lr=0.001)
    acc = train_loop(model, loaders, optimizer, epochs=5)



train_network()
