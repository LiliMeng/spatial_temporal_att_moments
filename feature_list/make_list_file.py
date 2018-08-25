"""
make feature and name list txt file for the moments data loader
Author: Lili Meng (menglili@cs.ubc.ca)
Date: August 25, 2018
"""


import numpy as np
import os


feature_dir = "/media/lili/fce9875a-a5c8-4c35-8f60-db60be29ea5d/extracted_features_moments_raw_hello/feature_val"
txt_file = open("feature_val_list.txt", mode='a')

txt_file.write("Feature"+"\n")


for subdir in sorted(os.listdir(feature_dir)):
	print("subdir :", subdir)

	for video in sorted(os.listdir(os.path.join(feature_dir, subdir))):
		if '.npy' in video:
			print("video: ", video)
			feature_name = os.path.join(subdir, video)
			print("feature_name: ", feature_name)
			txt_file.write(feature_name+"\n")


