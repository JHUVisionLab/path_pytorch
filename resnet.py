import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import pdb


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
		   'resnet152']


model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def batch_image_normalize(images,  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
	batchsize, h, w = images.shape[0], images.shape[2], images.shape[3]
	device = images.device
	mean = torch.tensor(mean, device = device).view(1,3,1,1)
	#mean = torch.tensor(mean).expand(batchsize,3, h, w)

	std = torch.tensor(std, device = device).view(1,3,1,1)
	#std = torch.tensor(std).expand(batchsize,3,h,w)
	normalized = images.sub_(mean).div_(std)
	return normalized

def test_normalize():
	images = torch.zeros(4,3,1536, 2048)
	print(batch_image_normalize(images))

def tile_images_FP(images):

	if images.shape[2] != 1536 or images.shape[3] != 2048:
		raise ValueError('Image to be tiled was not 1536x2048, instead it was: '
					 + images.shape[1] + 'x' + images.shape[2])

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

def _tile_base(image):
	# pad = torch.stack([0,0],[0,32],[0,192],[0,0]])
	# image = torch.pad(image, pad, 'CONSTANT')
	tile_base = F.interpolate(image, 224, mode = 'bilinear')
	return tile_base

def _tile_res1(image):
	#image = 1536 (H) x 2048 (W) --> 384 x 512

	"""
  	Tile at the coarser resolution (16x downsampled )
  
  	Args: 
    	image: Tensor of shape [1,1536, 2048,3]
  
  	Returns: 
    	Tensor of shape [4*3,3, 224,224]
  	"""
	image = F.interpolate(image, [384, 512], mode = 'bilinear')
	pad = (0,48,64,0) #left, right, top, bottom
	image = F.pad(image, pad, mode = "constant")
	size = 224
	stride = 112
	tiles = image.unfold(2, size, stride).unfold(3, size, stride)
	tiles = tiles.contiguous().view(-1, 3, 224, 224)
	return tiles


def _tile_res2(image):
	#fine resolution
	#image = 1536 (H) x 2048 (W)
	pad = (0,80,32,0) #left, right, top, bottom
	image = F.pad(image, pad, mode = "constant")
	size = 224
	stride = 112
	tiles = image.unfold(2, size, stride).unfold(3, size, stride)
	tiles = tiles.contiguous().view(-1, 3, 224, 224)
	return tiles

def _max_tile_3res(results, num_images):
  """
  Finds the max features for the different resolutions

  Args: [num_images*(18*13+4*3+1),1,1,2048]
  Returns: [num_images,1,1,6144]
  """
  list_images = list(torch.chunk(results, num_images,0))
  del results
  counter=0
  for im in list_images:
    res_base, res1, res2 = torch.split(im,[1,12,234],0) #hardcoded
    max1, _ = torch.max(res1, dim=0, keepdim=True)
    max2, _ = torch.max(res2, dim=0, keepdim=True)
    list_images[counter] = torch.cat([res_base,max1,max2,],3)
    counter += 1

  return torch.cat(list_images,0)

def tiling_test():
	images = torch.rand(4, 3, 1536, 2048)
	tiles = tile_images_FP(images)
	print(tiles.shape)

	features = torch.rand(4*(18*13+4*3+1),1,1,2048)
	max_feat = _max_tile_3res(features, 4)
	print(max_feat.shape)

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1000):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

class ResNet_2fc(nn.Module):

	def __init__(self, block, layers, num_classes=1000):
		self.inplanes = 64
		super(ResNet_2fc, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc1 = nn.Linear(512 * block.expansion, 512)
		self.dropout = nn.Dropout(p = 0.2)
		self.fc2 = nn.Linear(512, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, std=0.01)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		x = self.dropout(x)
		x = F.relu(x)
		x = self.fc2(x)

		return x

class ResNet_Tiling(nn.Module):

	def __init__(self, block, layers, num_classes=1000):
		self.inplanes = 64
		super(ResNet_Tiling, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc1 = nn.Linear(512 * block.expansion * 3, 512)
		self.dropout = nn.Dropout(p = 0.2)
		self.fc2 = nn.Linear(512, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, std=0.01)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		num_images = x.shape[0]
		x = tile_images_FP(x)
		x = batch_image_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = _max_tile_3res(x, num_images)
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		x = self.dropout(x)
		x = F.relu(x)
		x = self.fc2(x)

		return x

def resnet18(pretrained=False, **kwargs):
	"""Constructs a ResNet-18 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
	return model


def resnet34(pretrained=False, **kwargs):
	"""Constructs a ResNet-34 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
	return model


def resnet50(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
	
	return model


def resnet50_fc(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet_2fc(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict = False)

	  ### Freeze base model of resnet
	for param in model.parameters():
		param.requires_grad = False

	  ### Set fc layers to be trainabale
	model.fc1.weight.requires_grad = True
	model.fc1.bias.requires_grad = True
	model.fc2.weight.requires_grad = True
	model.fc2.bias.requires_grad = True
	return model

def resnet50_tiling(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet_Tiling(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict = False)

	  ### Freeze base model of resnet
	for param in model.parameters():
		param.requires_grad = False

	  ### Set fc layers to be trainabale
	model.fc1.weight.requires_grad = True
	model.fc1.bias.requires_grad = True
	model.fc2.weight.requires_grad = True
	model.fc2.bias.requires_grad = True
	return model


def resnet101(pretrained=False, **kwargs):
	"""Constructs a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
	return model


def resnet152(pretrained=False, **kwargs):
	"""Constructs a ResNet-152 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
	return model

