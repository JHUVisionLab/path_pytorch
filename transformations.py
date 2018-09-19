from torchvision import transforms
def multiresize():
	resize_list = [transforms.Resize([224, 224]), transforms.RandomCrop([224, 224]),transforms.RandomResizedCrop(size = 224, scale = (0.4,1)), transforms.CenterCrop([224,224])]
	random_apply_list = [transforms.ColorJitter()]
	transformation = transforms.Compose([transforms.Resize([1024,748]),
										transforms.RandomChoice(resize_list),
										transforms.RandomApply(random_apply_list),
										transforms.RandomVerticalFlip(),
										transforms.RandomHorizontalFlip(),
										transforms.ToTensor(),
										transforms.Normalize(mean=[0.485, 0.456, 0.406],
															std=[0.229, 0.224, 0.225])]
										)
	return transformation

def randomcrop_resize():
	resize_list = [transforms.Resize([224, 224]), transforms.RandomResizedCrop(size = 224, scale = (0.4,1))]
	random_apply_list = [transforms.ColorJitter()]
	transformation = transforms.Compose([transforms.RandomChoice(resize_list),
										transforms.RandomApply(random_apply_list),
										transforms.RandomVerticalFlip(),
										transforms.RandomHorizontalFlip(),
										transforms.ToTensor(),
										transforms.Normalize(mean=[0.485, 0.456, 0.406],
															std=[0.229, 0.224, 0.225])]
										)
	return transformation


def singleresize():
	random_apply_list = [transforms.ColorJitter()]
	transformation = transforms.Compose([transforms.Resize([224, 224]),
										 transforms.RandomApply(random_apply_list),
										 transforms.RandomVerticalFlip(),
										 transforms.RandomHorizontalFlip(),
										 transforms.ToTensor(),
										 transforms.Normalize(mean=[0.485, 0.456, 0.406],
															  std=[0.229, 0.224, 0.225])]
										)
	return transformation

def val():
	transformation = transforms.Compose([transforms.Resize([224, 224]),
											 transforms.ToTensor(),
											 transforms.Normalize(mean=[0.485, 0.456, 0.406],
															std=[0.229, 0.224, 0.225])
											])
	return transformation

def tiling_train():
	random_apply_list = [transforms.ColorJitter()]
	transformation = transforms.Compose([transforms.RandomApply(random_apply_list),
										transforms.RandomVerticalFlip(),
										transforms.RandomHorizontalFlip(),
										transforms.ToTensor(),
										transforms.Normalize(mean=[0.485, 0.456, 0.406],
															std=[0.229, 0.224, 0.225])])
	return transformation

def tiling_val():
	transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
															std=[0.229, 0.224, 0.225])])
	return transformation



