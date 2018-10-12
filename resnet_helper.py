import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pdb

def batch_image_normalize(images,  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
	""" Normalization of each tile instead of global image - not currently used"""
	batchsize, h, w = images.shape[0], images.shape[2], images.shape[3]
	device = images.device
	mean = torch.tensor(mean, device = device).view(1,3,1,1)
	std = torch.tensor(std, device = device).view(1,3,1,1)
	normalized = images.sub_(mean).div_(std)
	return normalized

def test_normalize():
	""" Test function for batch_image_normalize """
	mean=[0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]
	images = torch.zeros(4,3,1536, 2048)
	normalized = batch_image_normalize(images, mean, std)
	print(normalized.shape)

def tile_images_FP(images):
	""" Tile image in a feature pyramid like setup - 3 resolutions (including full image) """
	if images.shape[2] != 1536 or images.shape[3] != 2048:
		raise ValueError('Image to be tiled was not 1536x2048, instead it was: '
					 + images.shape[2] + 'x' + images.shape[3])

	num_images = images.shape[0]
	im_list = list(torch.chunk(images,num_images,0))
	del images
	counter=0
	for im in im_list:
		tile_base = _tile_base(im)
		tiles1 = _tile_res1(im)
		tiles2 = _tile_res2(im)
		im_list[counter] = torch.cat([tile_base,tiles1,tiles2],0)
		counter+=1
  
	return torch.cat(im_list,0)

def tile_images_2res(images):
	""" Tile image in a feature pyramid like setup - 2 resolutions """
	if images.shape[2] != 1536 or images.shape[3] != 2048:
		raise ValueError('Image to be tiled was not 1536x2048, instead it was: '
					 + images.shape[2] + 'x' + images.shape[3])

	num_images = images.shape[0]
	im_list = list(torch.chunk(images,num_images,0))
	del images
	counter=0
	for im in im_list:
		tiles1 = _tile_res1(im)
		tiles2 = _tile_res2(im)
		im_list[counter] = torch.cat([tiles1,tiles2],0)
		counter+=1
  
	return torch.cat(im_list,0)

def _tile_base(image):
	#image = 1536 (H) x 2048 (W) --> 224 x 224
	# pad = torch.stack([0,0],[0,32],[0,192],[0,0]])
	# image = torch.pad(image, pad, 'CONSTANT')
	tile_base = F.interpolate(image, 224, mode = 'bilinear')
	
	return tile_base

def _tile_res1(image):
	#image = 1536 (H) x 2048 (W) --> 384 x 512 --> 224 x 224

	"""
	Tile at the coarser resolution (16x downsampled )
  
	Args: 
		image: Tensor of shape [1,3, 1536, 2048]
  
	Returns: 
		Tensor of shape [4*3,3, 224,224]
	"""
	image = F.interpolate(image, [384, 512], mode = 'bilinear')
	pad = (0,48,64,0) #left, right, top, bottom
	image = F.pad(image, pad, mode = "constant")
	channels = list(torch.chunk(image,3,1))
	size = 224
	stride = 112
	counter = 0
	
	for channel in channels:
		tiles = channel.squeeze().unfold(0, size, stride).unfold(1, size, stride)
		tiles = tiles.contiguous().view(-1,1,224,224)
		channels[counter] = tiles
		counter+=1

	return torch.cat(channels,1)


def _tile_res2(image):
	#image = 1536 (H) x 2048 (W) --> 224 x 224 

	"""
	Tile at the fine resolution 
  
	Args: 
		image: Tensor of shape [1,3, 1536, 2048]
  
	Returns: 
		Tensor of shape [18*13,3, 224,224]
	"""
	pad = (0,80,32,0) #left, right, top, bottom
	image = F.pad(image, pad, mode = "constant")
	channels = list(torch.chunk(image,3,1))
	size = 224
	stride = 112
	counter = 0
	
	for channel in channels:
		tiles = channel.squeeze().unfold(0, size, stride).unfold(1, size, stride)
		tiles = tiles.contiguous().view(-1,1,224,224)
		channels[counter] = tiles
		counter+=1

	return torch.cat(channels,1)

def _max_tile_3res(results, num_images):
	"""
	Finds the max features for the different resolutions

	Args: 
		[num_images*(18*13+4*3+1),1,1,2048] if pooling before fc classification 
		[num_images*(18*13+4*3+1),1,1,4)] if pooling after fc classification 
	
	Returns: 
		[num_images,1,1,6144] if pooling before fc classification
		[num_images,4] if pooling after fc classification 

	"""
	list_images = list(torch.chunk(results, num_images,0))
	del results
	counter=0
	for im in list_images:
		res_base, res1, res2 = torch.split(im,[1,12,234],0) #hardcoded
		pdb.set_trace()
		max1, _ = torch.max(res1, dim=0, keepdim=True)
		max2, _ = torch.max(res2, dim=0, keepdim=True)
		list_images[counter] = torch.cat([res_base,max1,max2],1)
		counter += 1

	return torch.cat(list_images,0)

def _max_tile_global(results, num_images):
	"""
	Finds the max features for the different resolutions

	Args: 
		[num_images*(18*13+4*3+1),1,1,2048] if pooling before fc classification 
		[num_images*(18*13+4*3+1),1,1,4)] if pooling after fc classification 
	
	Returns: 
		[num_images,1,1,6144] if pooling before fc classification
		[num_images,4] if pooling after fc classification 

	"""
	list_images = list(torch.chunk(results, num_images,0))
	del results
	counter=0
	for im in list_images:
		pdb.set_trace()
		max_logits, _ = torch.max(res1, dim=0, keepdim=True)
		list_images[counter] = max_logits
		counter += 1

	pdb.set_trace()
	return torch.cat(list_images,0)

def _max_tile_2res(results, num_images):
	"""
	Finds the max features for the different resolutions

	Args: [num_images*(18*13+4*3),1,1,2048]
	Returns: [num_images,1,1,4096]
	"""
	list_images = list(torch.chunk(results, num_images,0))
	del results
	counter=0
	for im in list_images:
		res1, res2 = torch.split(im,[12,234],0) #hardcoded
		max1, _ = torch.max(res1, dim=0, keepdim=True)
		max2, _ = torch.max(res2, dim=0, keepdim=True)
		list_images[counter] = torch.cat([max1,max2],1)
		counter += 1

	return torch.cat(list_images,0)


def tile_res1_test(image):
	image = F.interpolate(image, [5, 5], mode = 'bilinear')
	channels = list(torch.chunk(image,3,1))
	size = 2
	stride = 2
	counter = 0
	
	for channel in channels:
		tiles = channel.squeeze().unfold(0, size, stride).unfold(1, size, stride)
		tiles = tiles.contiguous().view(-1,1,2,2)
		channels[counter] = tiles
		counter+=1

	return torch.cat(channels,1)

def tile_res2_test(image):
	channels = list(torch.chunk(image,3,1))
	counter=0
	size = 2
	stride = 2
	for channel in channels:

		tiles = channel.squeeze().unfold(0, size, stride).unfold(1, size, stride)
		tiles = tiles.contiguous().view(-1,1,2,2)
		channels[counter] = tiles
		counter+=1

	return torch.cat(channels,1)
def tiling_test():
	import numpy as np
	images = np.arange(600,dtype=np.float64).reshape((2,3,10,10))
	pdb.set_trace()
	images = torch.from_numpy(images)
	
	im_list = list(torch.chunk(images, images.shape[0], dim=0))
	
	counter = 0
	for im in im_list:
		pdb.set_trace()
		tile_base = F.interpolate(im, 2, mode = 'bilinear')
		tiles1 = tile_res1_test(im)
		tiles2 = tile_res2_test(im)
		im_list[counter] = torch.cat([tile_base,tiles1,tiles2],0)
		counter+=1
  
	tiles = torch.cat(im_list,0)
	pdb.set_trace()


	image1 = torch.tensor(([1,2,2,3])).view(1,4)
	image2 = torch.tensor(([0,3,2,1])).view(1,4)
	image = torch.cat([image1,image2],dim=0)
	print()
	pdb.set_trace()