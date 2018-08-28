"""
make feature and name list txt file for the moments data loader
Author: Lili Meng (menglili@cs.ubc.ca)
Date: August 25, 2018
"""


import numpy as np
import os


feature_dir = "/media/lili/f9020c94-3607-46d2-bac8-696f0d445708/extracted_features_moments_raw/training_features"
txt_file = open("feature_train_list.txt", mode='a')

txt_file.write("Feature"+"\n")


for subdir in sorted(os.listdir(feature_dir)):
	print("subdir :", subdir)

	for video in sorted(os.listdir(os.path.join(feature_dir, subdir))):
		if '.npy' in video:
			print("video: ", video)
			feature_name = os.path.join(subdir, video)
			print("feature_name: ", feature_name)
			txt_file.write(feature_name+"\n")


