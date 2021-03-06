from __future__ import print_function, division
import os
import torch
import numpy as np
import math

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
from torch.utils.data import random_split
from torchvision import transforms, utils, models
from PIL import Image
from cross_validation import k_folds, k_folds_2

import pandas as pd
from tensorboardX import SummaryWriter


import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # useful stateless functions

import nets 
import transformations
from PathologyDataset import PathologyDataset
#### Settings 

USE_GPU = True
TILING = True
dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 10

print('using device:', device)

if TILING: 
	NUM_TRAIN = 360
	NUM_VAL = 40
	batch_size = 4
	EPOCH = 100
	learning_rate = 2e-4
	k = 10
	num_classes = 4

	transform_train = transformations.tiling_train()
	transform_val = transformations.tiling_val()

else: 
	NUM_TRAIN = 360
	NUM_VAL = 40
	batch_size = 32
	EPOCH = 200
	learning_rate = 1e-3
	k = 10
	num_classes = 4

	transform_train = transformations.randomcrop_resize()
	transform_val = transformations.val()


def check_accuracy(loader, model, train, cur_epoch = None, filename=None, writer = None):
	"""evalute model and report accuracy

	args:
		loader: pytorch dataloader
		model: pytorch module
		train (boolean): in training mode or not
		filename: name of result file
		writer: tensorboard writer object for logging 

	return:
		acc: accuracy of evaluation 
	"""
	num_correct = 0
	num_samples = 0
	total_loss = 0

	if not train:
		c0_list = np.empty((0,1), dtype = np.float32)
		c1_list = np.empty((0,1), dtype = np.float32)
		c2_list= np.empty((0,1), dtype = np.float32)
		c3_list = np.empty((0,1), dtype = np.float32)
		pred_list = np.empty((0,), dtype = np.uint8)
		y_list = np.empty((0,), dtype = np.uint8)
		eval_list = np.empty((0,), dtype = bool)

	model.eval()  # set model to evaluation mode
	
	if train:
		print('Checking accuracy on validation set')
	else:
		print('Final Evaluation') 

	with torch.no_grad():
		counter = 0
		for x, y in loader:
			counter += 1
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)
			scores = model(x)
			loss = F.cross_entropy(scores, y)
			total_loss += loss
			_, preds = scores.max(1)
			num_correct += (preds == y).sum()
			num_samples += preds.size(0)
			
			if not train:
				y = y.data.cpu().numpy()
				preds = preds.data.cpu().numpy()
				p = F.softmax(scores).data.cpu().numpy()
				c0, c1, c2, c3 = np.split(p, 4, axis = 1)
				c0_list = np.append(c0_list, c0)
				c1_list = np.append(c1_list, c1)
				c2_list = np.append(c2_list, c2)
				c3_list = np.append(c3_list, c3)

				y_list = np.append(y_list, y, axis = 0)
				pred_list = np.append(pred_list, preds, axis = 0)
				eval_list = np.append(eval_list, preds == y, axis = 0)

		acc = float(num_correct) / num_samples

		if train:
			writer.add_scalar('eval/loss', total_loss/counter, cur_epoch)
			writer.add_scalar('eval/acc', acc, cur_epoch)
		
		print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
		print()

		if not train:
			results_dict = {'p0': c0_list.squeeze(), 
						'p1': c1_list.squeeze(), 
						'p2': c2_list.squeeze(), 
						'p3': c3_list.squeeze(), 
						'label': y_list, 
						'pred': pred_list, 
						'eval': eval_list}
			results = pd.DataFrame.from_dict(results_dict)
			results.to_csv(filename, index = False)
		
		return acc


def train_loop(model, loaders, optimizer, epochs=10, filename=None, log_dir=None, writer = None, scheduler = None):
	writer = SummaryWriter(log_dir)
	"""
	Train a model on CIFAR-10 using the PyTorch Module API.
	
	Inputs:
	- model: A PyTorch Module giving the model to train.
	- optimizer: An Optimizer object we will use to train the model
	- epochs: (Optional) A Python integer giving the number of epochs to train for
	
	Returns: model accuracy after training, and prints model accuracy through out training
	"""

	# configure multi-gpu training
	if torch.cuda.device_count() > 1:
		print("using", torch.cuda.device_count(), "GPUs")
		model = nn.DataParallel(model)
	
	model = model.to(device=device)  # move the model parameters to GPU
	
	# set up training and eval data loader
	loader_train = loaders['train']
	# train_num = loader_train.dataset.__len__()
	# batch_size = loader_train.batch_size
	loader_val = loaders['val']

	print('training begins')
	print('base learning rate: ', learning_rate)

	for e in range(epochs):
		total_loss = 0
		counter = 0

		# use scheduler for learning rate deacy 
		if scheduler:
			adjust_learning_rate(optimizer, scheduler)


		for t, (x, y) in enumerate(loader_train):
			counter+=1
			model.train()  # put model to training mode

			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)

			scores = model(x)
			loss = F.cross_entropy(scores, y)
			total_loss+=loss

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
				print('Epoch %d of %d, Iteration %d, loss = %.4f' % (e, epochs-1, t, loss.item()))
				print()
		
		if writer: 
			writer.add_scalar('train/loss', total_loss/counter, e)
		
		acc = check_accuracy(loader_val, model, train=True, cur_epoch=e, filename=None, writer=writer)

	print()
	acc = check_accuracy(loader_val, model, train=False, filename=filename)
	return acc


def train_network(ssh = True, op = 'SGD'):
	if ssh:
		img_dir='/workspace/path_data/Part-A_Original'
		results_dir = '/workspace/results_pytorch'
	else:
		img_dir='/Users/admin/desktop/path_pytorch/Part-A_Original'
		results_dir = '/Users/admin/desktop/path_pytorch/results'

	path_data_train = PathologyDataset(csv_file='microscopy_ground_truth.csv', img_dir=img_dir, shuffle = True, transform=transform_train)
	path_data_val = PathologyDataset(csv_file='microscopy_ground_truth.csv', img_dir=img_dir, shuffle = False, transform=transform_val)

	if path_data_train.shuffle:
		path_data_val.img_ids = path_data_train.img_ids.copy()
		path_data_val.img_labels = path_data_train.img_labels.copy()

	# initialize acc vector for cv results 
	acc = np.zeros((k,))
	
	# fold counter
	counter = 0

	# k-fold eval
	for train_idx, test_idx in k_folds_2(n_splits = k):
		### tensor log directory
		log_dir = os.path.join(results_dir, 'results_' + str(counter))
		if not os.path.exists(log_dir):
			os.mkdir(log_dir)
		
		print('training and evaluating fold ', counter)
		### result file

		filename = os.path.join(results_dir, 'results_' + str(counter) + '.csv')
		
		### initialize data loaders
		loader_train = torch.utils.data.DataLoader(dataset = path_data_train, batch_size = batch_size, sampler = sampler.SubsetRandomSampler(train_idx),num_workers=4)
		loader_val = torch.utils.data.DataLoader(dataset = path_data_val, batch_size = batch_size, sampler = sampler.SubsetRandomSampler(test_idx), num_workers=4)
		loaders = {'train': loader_train, 'val': loader_val}
		### initialize model
		model = nets.resnet50_train_tiling(num_classes, res = [0,1,2], pool_after = False)
		print(model)
		print()

		for name, p in model.named_parameters():
			print(name, p.requires_grad)

		### initialize optimizer
		if op == 'RMSprop':
			optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),lr = learning_rate, momentum = 0.9, weight_decay = 0.0005, eps = 1.0)
		elif op == 'Adam':
			optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = learning_rate)
		elif op == 'SGD':
			optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr = learning_rate, momentum = 0.9, weight_decay = 0.0005)
		else:
			raise ValueError('Unsupported Optimizer: '
					 + optim)

		### Scheduler
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)

		### call training/eval
		acc[counter] = train_loop(model, loaders, optimizer, epochs=EPOCH, filename=filename, log_dir=log_dir, scheduler = scheduler)

		### update counter
		counter+=1
	
	print('k-fold CV accuracy: ', acc)
	print('final mean accuracy: ', np.mean(acc))


def adjust_learning_rate(optimizer, scheduler):
	scheduler.step()
	print('current learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])
	print()



def test_cv(dset1, dset2):
	# make sure the two datasets are identical in ids and labels
	assert np.all(np.equal(dset1.img_ids, dset2.img_ids))
	assert np.all(np.equal(dset1.img_labels, dset2.img_labels))




train_network()
