from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

def moments_category_dict(category_file_name):

	lines = [line.strip() for line in open(category_file_name).readlines()]

	moments_label_dict = {}
	index = 0
	for line in lines:
		category_name = line
		moments_label_dict[category_name] = index
		index+=1

	return moments_label_dict
		
class MomentsDataset(Dataset):
	def __init__(self, mode, feature_data_dir, name_data_dir, category_dict, csv_file, transform=None):
		self.mode = mode
		self.feature_data_dir = feature_data_dir
		self.name_data_dir = name_data_dir
		self.dataset = pd.read_csv(csv_file)
		self.category_dict = category_dict

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		feature_file = os.path.join(self.feature_data_dir, self.dataset['Feature'][idx])
		
		if self.mode == 'train':
			name_file = os.path.join(self.name_data_dir, self.dataset['Feature'][idx].replace("names", "features"))
		elif self.mode == 'test':
			name_file = os.path.join(self.name_data_dir, self.dataset['Feature'][idx].replace("features", "name"))
		else:
			raise Exception("no such mode exist, only train or test mode")
		
		feature_per_video = np.load(feature_file)
		name_per_video = np.load(name_file)
		label_per_video = self.category_dict[name_file.split('/')[-2]]

		sample = {'feature': feature_per_video, 'label': label_per_video}

		return sample, name_per_video.tolist()

def get_loader(feature_data_dir, name_data_dir, category_dict, csv_file, batch_size, mode='train', dataset='moments'):
	"""Build and return data loader."""

	shuffle = True if mode == 'train' else False

	if dataset == 'moments':
		dataset = MomentsDataset(mode, feature_data_dir, name_data_dir, category_dict, csv_file)

	data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

	return data_loader

if __name__ == '__main__':

	label_dict = moments_category_dict("./feature_list/category_moment.txt")
	val_feature_dir = "/media/lili/fce9875a-a5c8-4c35-8f60-db60be29ea5d/extracted_features_moments_raw/feature_val"
	val_name_dir = "/media/lili/fce9875a-a5c8-4c35-8f60-db60be29ea5d/extracted_features_moments_raw/name_val"
	val_csv_file = "./feature_list/feature_val_list.csv"
	val_batch_size = 10
	val_data_loader = get_loader(feature_data_dir = val_feature_dir,
									name_data_dir = val_name_dir, 
									category_dict = label_dict,
									csv_file = val_csv_file,
									batch_size = val_batch_size,
									mode = 'test', 
									dataset='moments')

	for i, (sample, batch_name) in enumerate(val_data_loader):
		batch_feature = sample['feature']
		batch_label = sample['label']
		print("batch_feature.shape: ", batch_feature.shape)
		print("batch_label.shape: ", batch_label.shape)
		print("batch_name: ", batch_name)

		break
	
	train_feature_dir = "/media/lili/f9020c94-3607-46d2-bac8-696f0d445708/extracted_features_moments_raw/training_features"
	train_name_dir = "/media/lili/f9020c94-3607-46d2-bac8-696f0d445708/extracted_features_moments_raw/training_names"
	train_csv_file = "./feature_list/feature_train_list.csv"
	train_batch_size = 16
	train_data_loader = get_loader(feature_data_dir = train_feature_dir,
									name_data_dir = train_name_dir, 
									category_dict = label_dict,
									csv_file = train_csv_file,
									batch_size = train_batch_size,
									mode = 'train', 
									dataset='moments')

	for i, (sample, batch_name) in enumerate(train_data_loader):
		batch_feature = sample['feature']
		batch_label = sample['label']
		print("batch_feature.shape: ", batch_feature.shape)
		print("batch_label.shape: ", batch_label.shape)
		print("batch_name: ", batch_name)

		break